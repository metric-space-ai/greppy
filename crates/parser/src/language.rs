//! Language registry. Maps language names and file extensions to tree-sitter
//! grammars.
//!
//! ~18 languages have a wired extractor: **Rust, Python, JavaScript,
//! TypeScript, TSX, Go, Java, Ruby, PHP, C, C++, C#, Kotlin, Swift, Scala,
//! Lua, Bash, Zig**. All emit definitions + cross-file **CALLS** and
//! **IMPORTS**; **Rust** additionally emits the full **TYPE_REF / USES /
//! IMPLEMENTS / signature** edge set (the other languages' richer edges are
//! follow-on work). Unrecognised extensions fall back to
//! [`Language::Unsupported`], which returns an explicit error from the
//! extraction entry point — never silently degraded output.

use std::path::Path;

/// A supported language. `Unsupported` is a sentinel that the extraction
/// entry points reject with a structured `Error::NotImplemented`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Rust,
    /// Python — the second fully-supported extraction language.
    Python,
    /// JavaScript (`.js`, `.jsx`, `.mjs`, `.cjs`). The tree-sitter-javascript
    /// grammar parses JSX, so `.jsx` shares this variant.
    JavaScript,
    /// TypeScript. `tsx` selects the TSX grammar (`.tsx`); otherwise the plain
    /// TypeScript grammar (`.ts`). The TSX grammar does not accept the `<T>x`
    /// type-assertion form, and the TypeScript grammar does not accept JSX, so
    /// the extension must choose the right one.
    TypeScript {
        tsx: bool,
    },
    /// Go (`.go`) — the fifth fully-supported extraction language.
    Go,
    /// Ruby (`.rb`) — the sixth fully-supported extraction language.
    Ruby,
    /// Java (`.java`) — the seventh fully-supported extraction language.
    Java,
    /// C (`.c`, `.h`). `.h` is ambiguous between C and C++; it defaults to C.
    C,
    /// C++ (`.cpp`, `.cc`, `.cxx`, `.hpp`, `.hh`).
    Cpp,
    /// C# (`.cs`) — onboarded purely via the data-driven [`crate::spec`] path.
    CSharp,
    /// PHP (`.php`) — onboarded purely via the data-driven path.
    Php,
    /// Bash (`.sh`, `.bash`) — onboarded purely via the data-driven path.
    Bash,
    /// Lua (`.lua`) — onboarded purely via the data-driven [`crate::spec`] path.
    Lua,
    /// Kotlin (`.kt`, `.kts`) — onboarded purely via the data-driven path.
    Kotlin,
    /// Scala (`.scala`, `.sc`) — onboarded purely via the data-driven path.
    Scala,
    /// Swift (`.swift`) — onboarded purely via the data-driven path.
    Swift,
    /// Zig (`.zig`) — onboarded purely via the data-driven path.
    Zig,
    /// R (`.r`, `.R`) — onboarded purely via the data-driven path.
    R,
    /// A language supplied by the parallel-safe registry (`src/langs/*.rs`),
    /// wrapping its self-contained [`crate::registry::LangDef`]. New languages
    /// are added this way — one file, no shared edits — instead of a new enum
    /// variant. Behaves exactly like a hand-wired variant at runtime.
    Registered(&'static crate::registry::LangDef),
    /// Sentinel; reserved for follow-on languages.
    Unsupported(&'static str),
}

impl Language {
    /// The display name for the language (used in error messages and
    /// qualified names).
    pub fn name(self) -> &'static str {
        match self {
            Language::Rust => "rust",
            Language::Python => "python",
            Language::JavaScript => "javascript",
            Language::TypeScript { .. } => "typescript",
            Language::Go => "go",
            Language::Ruby => "ruby",
            Language::Java => "java",
            Language::C => "c",
            Language::Cpp => "cpp",
            Language::CSharp => "csharp",
            Language::Php => "php",
            Language::Bash => "bash",
            Language::Lua => "lua",
            Language::Kotlin => "kotlin",
            Language::Scala => "scala",
            Language::Swift => "swift",
            Language::Zig => "zig",
            Language::R => "r",
            Language::Registered(d) => d.name,
            Language::Unsupported(s) => s,
        }
    }

    /// Return the tree-sitter grammar. `Unsupported` panics; callers
    /// must check `is_supported()` first.
    pub fn grammar(self) -> tree_sitter::Language {
        match self {
            Language::Rust => tree_sitter_rust::LANGUAGE.into(),
            Language::Python => tree_sitter_python::LANGUAGE.into(),
            Language::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            Language::TypeScript { tsx: false } => {
                tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()
            }
            Language::TypeScript { tsx: true } => tree_sitter_typescript::LANGUAGE_TSX.into(),
            Language::Go => tree_sitter_go::LANGUAGE.into(),
            Language::Ruby => tree_sitter_ruby::LANGUAGE.into(),
            Language::Java => tree_sitter_java::LANGUAGE.into(),
            Language::C => tree_sitter_c::LANGUAGE.into(),
            Language::Cpp => tree_sitter_cpp::LANGUAGE.into(),
            Language::CSharp => tree_sitter_c_sharp::LANGUAGE.into(),
            Language::Php => tree_sitter_php::LANGUAGE_PHP.into(),
            Language::Bash => tree_sitter_bash::LANGUAGE.into(),
            Language::Lua => tree_sitter_lua::LANGUAGE.into(),
            Language::Kotlin => tree_sitter_kotlin_ng::LANGUAGE.into(),
            Language::Scala => tree_sitter_scala::LANGUAGE.into(),
            Language::Swift => tree_sitter_swift::LANGUAGE.into(),
            Language::Zig => tree_sitter_zig::LANGUAGE.into(),
            Language::R => tree_sitter_r::LANGUAGE.into(),
            Language::Registered(d) => (d.grammar)(),
            Language::Unsupported(s) => panic!("unsupported language: {s}"),
        }
    }

    pub fn is_supported(self) -> bool {
        if matches!(self, Language::Registered(_)) {
            return true;
        }
        matches!(
            self,
            Language::Rust
                | Language::Python
                | Language::JavaScript
                | Language::TypeScript { .. }
                | Language::Go
                | Language::Ruby
                | Language::Java
                | Language::C
                | Language::Cpp
                | Language::CSharp
                | Language::Php
                | Language::Bash
                | Language::Lua
                | Language::Kotlin
                | Language::Scala
                | Language::Swift
                | Language::Zig
                | Language::R
        )
    }
}

/// The languages this crate currently supports. Used by tests and by
/// the `parser --supported` introspection command.
pub const SUPPORTED_LANGUAGES: &[Language] = &[
    Language::Rust,
    Language::Python,
    Language::JavaScript,
    Language::TypeScript { tsx: false },
    Language::TypeScript { tsx: true },
    Language::Go,
    Language::Ruby,
    Language::Java,
    Language::C,
    Language::Cpp,
    Language::CSharp,
    Language::Php,
    Language::Bash,
    Language::Lua,
    Language::Kotlin,
    Language::Scala,
    Language::Swift,
    Language::Zig,
    Language::R,
];

/// Map a file path to a [`Language`]. Returns
/// `Language::Unsupported("<detected-from-extension>")` for unknown
/// extensions so callers can produce a structured error rather than
/// silently degrading.
pub fn language_for_path(path: &Path) -> Language {
    // Parallel-safe registry first: a language added as a self-contained
    // `src/langs/<lang>.rs` file wins if it claims this path. The legacy
    // hand-wired table below stays as the fallback for the original 18.
    if let Some(d) = crate::registry::LangDef::for_path(path) {
        return Language::Registered(d);
    }
    match path.extension().and_then(|s| s.to_str()) {
        Some("rs") => Language::Rust,
        Some("py") => Language::Python,
        Some("js") | Some("jsx") | Some("mjs") | Some("cjs") => Language::JavaScript,
        Some("ts") => Language::TypeScript { tsx: false },
        Some("tsx") => Language::TypeScript { tsx: true },
        Some("go") => Language::Go,
        Some("rb") => Language::Ruby,
        Some("java") => Language::Java,
        // `.h` is ambiguous between C and C++; default to C (acceptable).
        Some("c") | Some("h") => Language::C,
        Some("cpp") | Some("cc") | Some("cxx") | Some("hpp") | Some("hh") => Language::Cpp,
        Some("cs") => Language::CSharp,
        Some("php") => Language::Php,
        Some("sh") | Some("bash") => Language::Bash,
        Some("lua") => Language::Lua,
        Some("kt") | Some("kts") => Language::Kotlin,
        Some("scala") | Some("sc") => Language::Scala,
        Some("swift") => Language::Swift,
        Some("zig") => Language::Zig,
        Some("r") | Some("R") => Language::R,
        Some(ext) => {
            Language::Unsupported(Box::leak(format!("file extension .{ext}").into_boxed_str()))
        }
        None => Language::Unsupported("no file extension"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_path_is_recognised() {
        let l = language_for_path(Path::new("src/lib.rs"));
        assert_eq!(l, Language::Rust);
        assert!(l.is_supported());
    }

    #[test]
    fn python_path_is_recognised() {
        let l = language_for_path(Path::new("app/main.py"));
        assert_eq!(l, Language::Python);
        assert!(l.is_supported());
    }

    #[test]
    fn javascript_paths_are_recognised() {
        for p in ["a.js", "a.jsx", "a.mjs", "a.cjs"] {
            let l = language_for_path(Path::new(p));
            assert_eq!(l, Language::JavaScript, "for {p}");
            assert!(l.is_supported());
        }
    }

    #[test]
    fn typescript_paths_are_recognised() {
        let ts = language_for_path(Path::new("a.ts"));
        assert_eq!(ts, Language::TypeScript { tsx: false });
        assert!(ts.is_supported());

        let tsx = language_for_path(Path::new("a.tsx"));
        assert_eq!(tsx, Language::TypeScript { tsx: true });
        assert!(tsx.is_supported());
    }

    #[test]
    fn go_path_is_recognised() {
        let l = language_for_path(Path::new("src/lib.go"));
        assert_eq!(l, Language::Go);
        assert!(l.is_supported());
    }

    #[test]
    fn ruby_path_is_recognised() {
        let l = language_for_path(Path::new("app/main.rb"));
        assert_eq!(l, Language::Ruby);
        assert!(l.is_supported());
    }

    #[test]
    fn tsx_uses_tsx_grammar_and_ts_uses_ts_grammar() {
        // The two TypeScript variants must resolve to different grammars.
        let ts = Language::TypeScript { tsx: false }.grammar();
        let tsx = Language::TypeScript { tsx: true }.grammar();
        assert_ne!(ts, tsx, "ts and tsx must use distinct grammars");
    }

    #[test]
    fn java_path_is_recognised() {
        let l = language_for_path(Path::new("src/Main.java"));
        assert_eq!(l, Language::Java);
        assert!(l.is_supported());
    }

    #[test]
    fn c_paths_are_recognised() {
        // `.c` is C; `.h` is ambiguous and defaults to C.
        for p in ["src/a.c", "src/a.h"] {
            let l = language_for_path(Path::new(p));
            assert_eq!(l, Language::C, "for {p}");
            assert!(l.is_supported());
        }
    }

    #[test]
    fn cpp_paths_are_recognised() {
        for p in ["a.cpp", "a.cc", "a.cxx", "a.hpp", "a.hh"] {
            let l = language_for_path(Path::new(p));
            assert_eq!(l, Language::Cpp, "for {p}");
            assert!(l.is_supported());
        }
    }

    #[test]
    fn unknown_path_is_unsupported() {
        let l = language_for_path(Path::new("src/lib.unknownext"));
        assert!(matches!(l, Language::Unsupported(_)));
        assert!(!l.is_supported());
    }

    #[test]
    fn no_extension_is_unsupported() {
        // A no-extension filename that no registry language claims. (`Makefile`
        // is now legitimately supported by the `make` registry language, so it
        // would no longer be a valid "unsupported" fixture.)
        let l = language_for_path(Path::new("COPYING"));
        assert!(matches!(l, Language::Unsupported(_)));
    }

    #[test]
    fn supported_languages_lists_all_languages() {
        assert_eq!(
            SUPPORTED_LANGUAGES,
            &[
                Language::Rust,
                Language::Python,
                Language::JavaScript,
                Language::TypeScript { tsx: false },
                Language::TypeScript { tsx: true },
                Language::Go,
                Language::Ruby,
                Language::Java,
                Language::C,
                Language::Cpp,
                Language::CSharp,
                Language::Php,
                Language::Bash,
                Language::Lua,
                Language::Kotlin,
                Language::Scala,
                Language::Swift,
                Language::Zig,
                Language::R,
            ]
        );
    }

    #[test]
    fn csharp_php_bash_paths_are_recognised() {
        for (p, want) in [
            ("Widget.cs", Language::CSharp),
            ("User.php", Language::Php),
            ("build.sh", Language::Bash),
            ("lib.bash", Language::Bash),
        ] {
            let l = language_for_path(Path::new(p));
            assert_eq!(l, want, "for {p}");
            assert!(l.is_supported());
        }
    }

    #[test]
    fn batch_onboarded_paths_are_recognised() {
        for (p, want) in [
            ("init.lua", Language::Lua),
            ("Widget.kt", Language::Kotlin),
            ("build.kts", Language::Kotlin),
            ("Widget.scala", Language::Scala),
            ("script.sc", Language::Scala),
            ("App.swift", Language::Swift),
            ("main.zig", Language::Zig),
            ("analysis.r", Language::R),
            ("analysis.R", Language::R),
        ] {
            let l = language_for_path(Path::new(p));
            assert_eq!(l, want, "for {p}");
            assert!(l.is_supported());
        }
    }
}
