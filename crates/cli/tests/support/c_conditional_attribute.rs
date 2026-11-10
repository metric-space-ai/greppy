// Shared std-only compiler fixtures. Compile this same file with rustc --test
// for the native oracle; the CLI and edit-unit tests import the identical bytes.
use std::process::{Command, Stdio};

// These bytes are shared by the real CLI check and the opt-in compiler oracle.
// GCC selects noipa; Clang selects noinline. Never infer one branch's validity
// from the other compiler, or erase an unchecked branch to accept the source.
pub const C_ATTRIBUTE_PREFIX: &str = "/* stable native observation boundary */\n#if defined(__GNUC__) && !defined(__clang__)\n__attribute__((noipa))\n#elif defined(__clang__)\n__attribute__((noinline))\n#endif\n";

pub fn c_attribute_cases() -> Vec<(&'static str, String, [bool; 2])> {
    let function = "int example(int n) { return n+1; }\n";
    vec![
        (
            "complete",
            format!("{C_ATTRIBUTE_PREFIX}{function}"),
            [true, true],
        ),
        (
            "complete-crlf",
            format!("{C_ATTRIBUTE_PREFIX}{function}").replace('\n', "\r\n"),
            [true, true],
        ),
        (
            "comment-lookalike",
            format!(
                "/*\n{}*/\n{function}",
                C_ATTRIBUTE_PREFIX.split_once('\n').unwrap().1
            ),
            [true, true],
        ),
        (
            "broken-body",
            format!("{C_ATTRIBUTE_PREFIX}int example(int n) {{ return n+; }}\n"),
            [false, false],
        ),
        (
            "broken-gcc-attribute",
            format!(
                "{}{function}",
                C_ATTRIBUTE_PREFIX.replace("__attribute__((noipa))", "__attribute__((noipa)")
            ),
            [false, true],
        ),
        (
            "broken-clang-attribute",
            format!(
                "{}{function}",
                C_ATTRIBUTE_PREFIX.replace("__attribute__((noinline))", "__attribute__((noinline)")
            ),
            [true, false],
        ),
        (
            "broken-following-declaration",
            format!("{C_ATTRIBUTE_PREFIX}{function}int broken = ;\n"),
            [false, false],
        ),
    ]
}

#[test]
#[ignore = "requires explicit GCC and Clang paths on the admitted build lane"]
fn c_conditional_attribute_compiler_controls() {
    let directory = std::env::temp_dir().join(format!(
        "greppy-c-attribute-compilers-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&directory).expect("create owned compiler fixture directory");
    struct OwnedDirectory(std::path::PathBuf);
    impl Drop for OwnedDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let fixture = OwnedDirectory(directory);
    for (index, variable) in ["GREPPY_C_ATTRIBUTE_GCC", "GREPPY_C_ATTRIBUTE_CLANG"]
        .iter()
        .enumerate()
    {
        let compiler = std::env::var_os(variable).expect("set the real compiler path");
        let version = Command::new(&compiler).arg("--version").output().unwrap();
        assert!(version.status.success());
        eprintln!(
            "{variable}={compiler:?}: {}",
            String::from_utf8_lossy(&version.stdout)
        );
        let macros = Command::new(&compiler)
            .args(["-dM", "-E", "-x", "c", "-"])
            .stdin(Stdio::null())
            .output()
            .unwrap();
        assert!(macros.status.success());
        let macros = String::from_utf8(macros.stdout).unwrap();
        assert!(macros.contains("#define __GNUC__ "));
        assert_eq!(macros.contains("#define __clang__ "), index == 1);
        for (name, source, expected) in c_attribute_cases() {
            let path = fixture.0.join(format!("{name}.c"));
            std::fs::write(&path, &source).unwrap();
            let mut command = Command::new(&compiler);
            command
                .args([
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-pedantic",
                    "-fsyntax-only",
                ])
                .arg(&path)
                .stdin(Stdio::null());
            eprintln!("compiler fixture={name} bytes={source:?} argv={command:?}");
            let output = command.output().unwrap();
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!("exit={:?} stderr={stderr}", output.status.code());
            assert_eq!(output.status.success(), expected[index], "{name}: {stderr}");
            if !expected[index] {
                assert!(stderr.contains("error:"), "{name}: {stderr}");
            }
        }
    }
}
