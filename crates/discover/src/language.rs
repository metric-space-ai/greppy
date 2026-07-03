//! Broad language detection (extension + special filenames).
//!
//! This mirrors upstream `src/discover/language.c`'s detection *breadth*:
//! it recognizes the same wide set of file extensions and well-known
//! filenames and returns a [`DetectedLanguage`]. Detection breadth is
//! intentionally decoupled from extraction support: the parser/extractor
//! in `crates/parser` currently only *extracts* Rust, but discovery
//! should still classify every file it sees so the indexer can record an
//! accurate `file_state` (e.g. "detected Go, extraction unsupported,
//! skipped") rather than treating non-Rust files as opaque.
//!
//! This registry lives in the discover crate (not the parser's
//! `Language` enum) on purpose: discovery owns *what a file is*, while
//! the parser owns *what we can extract*. Keeping them separate lets the
//! detection table grow to upstream breadth without forcing the
//! extractor's `Language` enum to enumerate languages it cannot parse.

use std::path::Path;

/// A language detected from a file's name or extension. This is a
/// detection-breadth enum: presence here does **not** imply the
/// extractor supports it. `Unknown` carries the raw extension (or empty)
/// so callers can log/record what was seen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectedLanguage {
    Bash,
    C,
    Cpp,
    CSharp,
    Clojure,
    CMake,
    Css,
    Dart,
    Dockerfile,
    Dotenv,
    Elixir,
    Elm,
    Erlang,
    FSharp,
    Fortran,
    Go,
    GoMod,
    GraphQl,
    Groovy,
    Haskell,
    Hcl,
    Html,
    Ini,
    Java,
    JavaScript,
    Json,
    Julia,
    Just,
    Kotlin,
    Kustomize,
    Lua,
    Makefile,
    Markdown,
    Matlab,
    Meson,
    Nix,
    OCaml,
    Perl,
    Php,
    Protobuf,
    Python,
    R,
    Requirements,
    Ruby,
    Rust,
    Scala,
    Scss,
    Sql,
    Starlark,
    Svelte,
    Swift,
    Toml,
    Tsx,
    TypeScript,
    Vue,
    Xml,
    Yaml,
    Zsh,
    /// No mapping. The `&str` is the lowercased extension without the
    /// dot, or `""` when the file had no extension and no special-name
    /// match.
    Unknown(&'static str),
}

impl DetectedLanguage {
    /// Stable lowercase name, used for logging and `file_state` records.
    /// `Unknown` reports the raw extension (or `"unknown"` when empty).
    pub fn name(self) -> &'static str {
        use DetectedLanguage::*;
        match self {
            Bash => "bash",
            C => "c",
            Cpp => "cpp",
            CSharp => "csharp",
            Clojure => "clojure",
            CMake => "cmake",
            Css => "css",
            Dart => "dart",
            Dockerfile => "dockerfile",
            Dotenv => "dotenv",
            Elixir => "elixir",
            Elm => "elm",
            Erlang => "erlang",
            FSharp => "fsharp",
            Fortran => "fortran",
            Go => "go",
            GoMod => "gomod",
            GraphQl => "graphql",
            Groovy => "groovy",
            Haskell => "haskell",
            Hcl => "hcl",
            Html => "html",
            Ini => "ini",
            Java => "java",
            JavaScript => "javascript",
            Json => "json",
            Julia => "julia",
            Just => "just",
            Kotlin => "kotlin",
            Kustomize => "kustomize",
            Lua => "lua",
            Makefile => "makefile",
            Markdown => "markdown",
            Matlab => "matlab",
            Meson => "meson",
            Nix => "nix",
            OCaml => "ocaml",
            Perl => "perl",
            Php => "php",
            Protobuf => "protobuf",
            Python => "python",
            R => "r",
            Requirements => "requirements",
            Ruby => "ruby",
            Rust => "rust",
            Scala => "scala",
            Scss => "scss",
            Sql => "sql",
            Starlark => "starlark",
            Svelte => "svelte",
            Swift => "swift",
            Toml => "toml",
            Tsx => "tsx",
            TypeScript => "typescript",
            Vue => "vue",
            Xml => "xml",
            Yaml => "yaml",
            Zsh => "zsh",
            Unknown("") => "unknown",
            Unknown(ext) => ext,
        }
    }

    /// True when the language was positively identified (i.e. not
    /// `Unknown`). Detection-positive does not imply extraction-supported.
    pub fn is_detected(self) -> bool {
        !matches!(self, DetectedLanguage::Unknown(_))
    }
}

/// Special filenames (exact basename match) -> language. Mirrors the
/// upstream `FILENAME_TABLE`. Checked before the extension table because
/// e.g. `Makefile` has no extension and `go.mod` would otherwise map via
/// `.mod` (which is not in the ext table anyway).
fn language_for_filename(name: &str) -> Option<DetectedLanguage> {
    use DetectedLanguage::*;
    Some(match name {
        "CMakeLists.txt" => CMake,
        "Dockerfile" => Dockerfile,
        "GNUmakefile" | "Makefile" | "makefile" => Makefile,
        "meson.build" | "meson.options" | "meson_options.txt" => Meson,
        "kustomization.yaml" | "kustomization.yml" => Kustomize,
        ".zshrc" | ".zshenv" | ".zprofile" => Zsh,
        "justfile" | "Justfile" | ".justfile" => Just,
        "BUILD" | "BUILD.bazel" | "WORKSPACE" | "WORKSPACE.bazel" => Starlark,
        "requirements.txt" | "requirements-dev.txt" | "requirements-test.txt" => Requirements,
        "go.mod" => GoMod,
        ".env" | ".env.local" => Dotenv,
        _ => return None,
    })
}

/// Lowercased-extension -> language. Mirrors the upstream `EXT_TABLE`
/// breadth for the mainstream and common-secondary languages. The match
/// is on the extension **without** the leading dot, lowercased.
///
/// Note: `.m` maps to MATLAB here (the upstream default before content
/// disambiguation between Objective-C / Magma / MATLAB, which is a
/// separate follow-on task — see PORT_COMPLETENESS discover breakdown).
fn language_for_ext(ext_lower: &str) -> Option<DetectedLanguage> {
    use DetectedLanguage::*;
    Some(match ext_lower {
        "bash" | "sh" => Bash,
        "c" => C,
        "cc" | "ccm" | "cpp" | "cppm" | "cxx" | "h" | "hh" | "hpp" | "hxx" | "ixx" => Cpp,
        "cs" => CSharp,
        "clj" | "cljc" | "cljs" => Clojure,
        "cmake" => CMake,
        "css" => Css,
        "dart" => Dart,
        "dockerfile" => Dockerfile,
        "env" => Dotenv,
        "ex" | "exs" => Elixir,
        "elm" => Elm,
        "erl" => Erlang,
        "fs" | "fsi" | "fsx" => FSharp,
        "f03" | "f08" | "f90" | "f95" => Fortran,
        "go" => Go,
        "gql" | "graphql" => GraphQl,
        "gradle" | "groovy" => Groovy,
        "hs" => Haskell,
        "hcl" | "tf" => Hcl,
        "htm" | "html" => Html,
        "cfg" | "conf" | "ini" => Ini,
        "java" => Java,
        "js" | "jsx" | "mjs" | "cjs" => JavaScript,
        "json" => Json,
        "jl" => Julia,
        "just" | "justfile" => Just,
        "kt" | "kts" => Kotlin,
        "lua" => Lua,
        "mk" => Makefile,
        "md" | "mdx" => Markdown,
        "m" | "matlab" | "mlx" => Matlab,
        "meson" => Meson,
        "nix" => Nix,
        "ml" | "mli" => OCaml,
        "pl" | "pm" => Perl,
        "php" => Php,
        "proto" => Protobuf,
        "py" => Python,
        "r" => R,
        "gemspec" | "rake" | "rb" => Ruby,
        "rs" => Rust,
        "sc" | "scala" => Scala,
        "scss" => Scss,
        "sql" => Sql,
        "bzl" | "star" => Starlark,
        "svelte" => Svelte,
        "swift" => Swift,
        "toml" => Toml,
        "tsx" => Tsx,
        "ts" | "mts" | "cts" => TypeScript,
        "vue" => Vue,
        "xml" | "xsd" | "xsl" | "svg" => Xml,
        "yaml" | "yml" => Yaml,
        "zsh" => Zsh,
        _ => return None,
    })
}

/// Detect a [`DetectedLanguage`] for `path` by basename then extension.
///
/// Resolution order (matching upstream): exact special-filename match
/// first, then the extension table, then `Unknown`. The extension is
/// lowercased so `.PY`/`.py` and `.R`/`.r` both resolve. Returns
/// `Unknown("")` for a file with neither a recognized name nor a
/// recognized extension.
pub fn detect_language(path: &Path) -> DetectedLanguage {
    if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
        if let Some(lang) = language_for_filename(name) {
            return lang;
        }
    }
    match path.extension().and_then(|s| s.to_str()) {
        Some(ext) => {
            let lower = ext.to_ascii_lowercase();
            match language_for_ext(&lower) {
                Some(lang) => lang,
                None => DetectedLanguage::Unknown(intern_ext(&lower)),
            }
        }
        None => DetectedLanguage::Unknown(""),
    }
}

/// Map a script interpreter basename (e.g. `python3`, `bash`, `node`)
/// to a [`DetectedLanguage`]. This is the shebang counterpart to the
/// extension table: many scripts (CI helpers, hooks, build glue) ship
/// with no extension and only a `#!` line, so extension-only detection
/// reports them as `Unknown`. Mirrors the interpreter set upstream
/// `lang_specs` recognizes for executable scripts.
///
/// The basename is matched after stripping a trailing version suffix
/// (`python3.11` -> `python`) and is compared lowercased.
fn language_for_interpreter(interp: &str) -> Option<DetectedLanguage> {
    use DetectedLanguage::*;
    // Strip a trailing version like `python3` / `python3.11` / `ruby2.7`
    // down to the alphabetic stem so versioned interpreters resolve.
    let stem: String = interp
        .chars()
        .take_while(|c| c.is_ascii_alphabetic() || *c == '_' || *c == '-')
        .collect();
    let stem = stem.to_ascii_lowercase();
    Some(match stem.as_str() {
        "sh" | "bash" | "dash" | "ash" | "ksh" => Bash,
        "zsh" => Zsh,
        "python" | "python_" | "py" | "pypy" => Python,
        "ruby" => Ruby,
        "perl" => Perl,
        "node" | "nodejs" | "deno" | "bun" => JavaScript,
        "lua" | "luajit" => Lua,
        "php" => Php,
        "r" | "rscript" => R,
        "julia" => Julia,
        "groovy" => Groovy,
        "fish" => Bash,
        _ => return None,
    })
}

/// Parse the interpreter from a `#!` shebang line and resolve it to a
/// language. Returns `None` when the bytes are not a shebang or the
/// interpreter is unrecognized.
///
/// Handles both direct (`#!/bin/bash`) and `env`-style
/// (`#!/usr/bin/env python3 -u`) shebangs: with `env`, the *next* token
/// after `env` (skipping `-S`/`-i`/`VAR=...` style args) is the
/// interpreter. Only the first line is consulted, and only its leading
/// portion (a shebang must be the very first bytes of the file).
pub fn language_from_shebang(first_bytes: &[u8]) -> Option<DetectedLanguage> {
    if !first_bytes.starts_with(b"#!") {
        return None;
    }
    // Take the first line only.
    let line_end = first_bytes
        .iter()
        .position(|&b| b == b'\n' || b == b'\r')
        .unwrap_or(first_bytes.len());
    let line = std::str::from_utf8(&first_bytes[2..line_end]).ok()?;
    let mut tokens = line.split_whitespace();
    let first = tokens.next()?;
    // Basename of the first token (the program path).
    let prog = first.rsplit('/').next().unwrap_or(first);
    if prog == "env" {
        // Skip env flags / VAR=VALUE assignments to find the interpreter.
        for tok in tokens {
            if tok.starts_with('-') || tok.contains('=') {
                continue;
            }
            let base = tok.rsplit('/').next().unwrap_or(tok);
            return language_for_interpreter(base);
        }
        None
    } else {
        language_for_interpreter(prog)
    }
}

/// Number of leading bytes read from a file when sniffing a shebang.
/// A shebang line is short; 256 bytes is ample and bounds the read.
const SHEBANG_SNIFF_BYTES: usize = 256;

/// Detect a language for `path`, falling back to a `#!` shebang sniff
/// when the name/extension yield no positive match.
///
/// Resolution order: special-filename, then extension (both via
/// [`detect_language`]); if that is `Unknown` *and* the file begins with
/// a recognized shebang, the interpreter's language is returned. This
/// catches extensionless scripts (e.g. a `hooks/pre-commit` that is a
/// bash script, or a `tools/gen` that is a Python script). Reads at most
/// [`SHEBANG_SNIFF_BYTES`]; on any read error it returns the
/// name/extension result unchanged.
pub fn detect_language_with_shebang(path: &Path) -> DetectedLanguage {
    let by_name = detect_language(path);
    if by_name.is_detected() {
        return by_name;
    }
    use std::io::Read;
    let mut f = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return by_name,
    };
    let mut buf = [0u8; SHEBANG_SNIFF_BYTES];
    let n = match f.read(&mut buf) {
        Ok(n) => n,
        Err(_) => return by_name,
    };
    language_from_shebang(&buf[..n]).unwrap_or(by_name)
}

/// Intern a small fixed set of commonly-seen unknown extensions to a
/// `&'static str` without leaking memory on the hot path. For anything
/// outside the set we fall back to `""` (the name reported is then
/// `"unknown"`), which is acceptable for a `file_state` log: the caller
/// still has the full path. We deliberately do NOT `Box::leak` arbitrary
/// extensions here — an attacker-controlled tree with thousands of
/// distinct junk extensions must not be able to grow process memory
/// unboundedly.
fn intern_ext(ext_lower: &str) -> &'static str {
    match ext_lower {
        "txt" => "txt",
        "lock" => "lock",
        "log" => "log",
        "csv" => "csv",
        "tsv" => "tsv",
        "cfg" => "cfg",
        "bin" => "bin",
        "dat" => "dat",
        "o" => "o",
        "a" => "a",
        "so" => "so",
        "png" => "png",
        "jpg" => "jpg",
        "pdf" => "pdf",
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_extension() {
        assert_eq!(
            detect_language(Path::new("src/lib.rs")),
            DetectedLanguage::Rust
        );
        assert!(detect_language(Path::new("a.rs")).is_detected());
    }

    #[test]
    fn breadth_mainstream_languages() {
        let cases = [
            ("main.go", DetectedLanguage::Go),
            ("app.py", DetectedLanguage::Python),
            ("index.js", DetectedLanguage::JavaScript),
            ("index.jsx", DetectedLanguage::JavaScript),
            ("app.ts", DetectedLanguage::TypeScript),
            ("app.tsx", DetectedLanguage::Tsx),
            ("Main.java", DetectedLanguage::Java),
            ("a.cpp", DetectedLanguage::Cpp),
            ("a.hpp", DetectedLanguage::Cpp),
            ("a.c", DetectedLanguage::C),
            ("a.rb", DetectedLanguage::Ruby),
            ("a.kt", DetectedLanguage::Kotlin),
            ("a.swift", DetectedLanguage::Swift),
            ("a.scala", DetectedLanguage::Scala),
            ("a.cs", DetectedLanguage::CSharp),
            ("a.php", DetectedLanguage::Php),
            ("Cargo.toml", DetectedLanguage::Toml),
            ("config.yaml", DetectedLanguage::Yaml),
            ("config.yml", DetectedLanguage::Yaml),
            ("data.json", DetectedLanguage::Json),
            ("main.tf", DetectedLanguage::Hcl),
        ];
        for (p, want) in cases {
            assert_eq!(detect_language(Path::new(p)), want, "path {p}");
        }
    }

    #[test]
    fn extension_is_case_insensitive() {
        assert_eq!(detect_language(Path::new("A.PY")), DetectedLanguage::Python);
        assert_eq!(detect_language(Path::new("script.R")), DetectedLanguage::R);
        assert_eq!(detect_language(Path::new("LIB.RS")), DetectedLanguage::Rust);
    }

    #[test]
    fn special_filenames() {
        let cases = [
            ("Makefile", DetectedLanguage::Makefile),
            ("GNUmakefile", DetectedLanguage::Makefile),
            ("CMakeLists.txt", DetectedLanguage::CMake),
            ("Dockerfile", DetectedLanguage::Dockerfile),
            ("go.mod", DetectedLanguage::GoMod),
            ("requirements.txt", DetectedLanguage::Requirements),
            ("kustomization.yaml", DetectedLanguage::Kustomize),
            ("BUILD.bazel", DetectedLanguage::Starlark),
            (".zshrc", DetectedLanguage::Zsh),
        ];
        for (p, want) in cases {
            assert_eq!(detect_language(Path::new(p)), want, "filename {p}");
        }
    }

    #[test]
    fn special_filename_beats_extension() {
        // CMakeLists.txt would map via `.txt` (unknown) if the ext table
        // ran first; the filename table must win.
        assert_eq!(
            detect_language(Path::new("a/b/CMakeLists.txt")),
            DetectedLanguage::CMake
        );
    }

    #[test]
    fn unknown_extension_reports_ext_name() {
        let l = detect_language(Path::new("notes.txt"));
        assert_eq!(l, DetectedLanguage::Unknown("txt"));
        assert!(!l.is_detected());
        assert_eq!(l.name(), "txt");
    }

    #[test]
    fn no_extension_is_unknown_empty() {
        let l = detect_language(Path::new("READMEFILE"));
        assert_eq!(l, DetectedLanguage::Unknown(""));
        assert_eq!(l.name(), "unknown");
    }

    #[test]
    fn unbounded_junk_extension_does_not_leak() {
        // Extensions outside the interned set collapse to "" rather than
        // leaking a new &'static str per distinct extension.
        let l = detect_language(Path::new("x.zzqqweirdext"));
        assert_eq!(l, DetectedLanguage::Unknown(""));
    }

    #[test]
    fn m_extension_defaults_to_matlab() {
        // Upstream default before content disambiguation.
        assert_eq!(
            detect_language(Path::new("solve.m")),
            DetectedLanguage::Matlab
        );
    }

    #[test]
    fn shebang_direct_interpreters() {
        let cases = [
            (&b"#!/bin/bash\necho hi\n"[..], DetectedLanguage::Bash),
            (&b"#!/bin/sh\n"[..], DetectedLanguage::Bash),
            (&b"#!/usr/bin/zsh\n"[..], DetectedLanguage::Zsh),
            (&b"#!/usr/bin/perl\n"[..], DetectedLanguage::Perl),
            (&b"#!/usr/local/bin/lua\n"[..], DetectedLanguage::Lua),
        ];
        for (bytes, want) in cases {
            assert_eq!(language_from_shebang(bytes), Some(want), "{bytes:?}");
        }
    }

    #[test]
    fn shebang_env_style_and_versioned() {
        assert_eq!(
            language_from_shebang(b"#!/usr/bin/env python3\n"),
            Some(DetectedLanguage::Python)
        );
        assert_eq!(
            language_from_shebang(b"#!/usr/bin/env python3.11 -u\n"),
            Some(DetectedLanguage::Python)
        );
        assert_eq!(
            language_from_shebang(b"#!/usr/bin/env node\n"),
            Some(DetectedLanguage::JavaScript)
        );
        assert_eq!(
            language_from_shebang(b"#!/usr/bin/env -S ruby2.7\n"),
            Some(DetectedLanguage::Ruby)
        );
    }

    #[test]
    fn shebang_unknown_or_absent() {
        assert_eq!(language_from_shebang(b"not a shebang\n"), None);
        assert_eq!(language_from_shebang(b"#!/bin/totally-made-up\n"), None);
        assert_eq!(language_from_shebang(b""), None);
        // A `#` comment that is not a shebang must not match.
        assert_eq!(language_from_shebang(b"# python\n"), None);
    }

    #[test]
    fn detect_with_shebang_reads_extensionless_script() {
        let dir = crate::tests_support::unique_tmp();
        let script = dir.join("pre-commit");
        std::fs::write(&script, b"#!/usr/bin/env python3\nprint('hi')\n").unwrap();
        // No extension => extension detection is Unknown; shebang rescues.
        assert_eq!(detect_language(&script), DetectedLanguage::Unknown(""));
        assert_eq!(
            detect_language_with_shebang(&script),
            DetectedLanguage::Python
        );
    }

    #[test]
    fn detect_with_shebang_prefers_extension_when_known() {
        // A .rs file with a misleading shebang must still be Rust: the
        // name/extension match wins and we never read the file.
        let dir = crate::tests_support::unique_tmp();
        let f = dir.join("main.rs");
        std::fs::write(&f, b"#!/usr/bin/env python3\nfn main(){}\n").unwrap();
        assert_eq!(detect_language_with_shebang(&f), DetectedLanguage::Rust);
    }

    #[test]
    fn detect_with_shebang_unknown_stays_unknown() {
        let dir = crate::tests_support::unique_tmp();
        let f = dir.join("data");
        std::fs::write(&f, b"plain content, no shebang\n").unwrap();
        assert_eq!(
            detect_language_with_shebang(&f),
            DetectedLanguage::Unknown("")
        );
    }
}
