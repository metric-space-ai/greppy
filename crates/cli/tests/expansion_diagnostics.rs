//! Refusal guidance must not recommend unsupported expansion flags.
use std::process::Command;

#[test]
fn expand_line_range_refusal_names_real_pagination_without_inventing_flags() {
    for arguments in [
        vec!["expand", "d3e35587a7c9b0d2", "--lines", "1093:1110"],
        vec!["expand", "d3e35587a7c9b0d2", "--line", "1093"],
        vec!["expand", "d3e35587a7c9b0d2", "--lines=1093:1110"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_greppy"))
            .args(&arguments)
            .output()
            .expect("run expansion refusal");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.status.code(), Some(64), "{arguments:?}: {text}");
        assert!(text.contains("prepared evidence page"), "{text}");
        assert!(text.contains("greppy expand ID --json"), "{text}");
        assert!(text.contains("next.command"), "{text}");
        assert!(text.contains("greppy read-file PATH --lines A:B"), "{text}");
        assert!(!text.contains("--offset"), "{text}");
        assert!(!text.contains("--limit"), "{text}");
    }
}
