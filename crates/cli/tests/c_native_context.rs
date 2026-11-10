#![cfg(unix)]
use std::{
    fs,
    io::Write,
    path::PathBuf,
    process::{Command, Output, Stdio},
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
const SOURCE:&str="#include \"config.h\"\nint choose(int condition) {\n    (void)condition;\n#if OPTION\n    if (condition) { return 1; } else\n#endif\n    { return VALUE; }\n}\n";
struct Fixture {
    _temp: tempfile::TempDir,
    root: PathBuf,
    store: PathBuf,
    compiler: String,
    family: String,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::Builder::new()
            .prefix("greppy-c-native-contract-")
            .tempdir()
            .unwrap();
        let root = temp.path().join("repo");
        fs::create_dir_all(root.join(".git")).unwrap();
        let compiler =
            std::env::var("GREPPY_TEST_C_COMPILER").unwrap_or_else(|_| "/usr/bin/gcc".into());
        let version = Command::new(&compiler)
            .arg("--version")
            .output()
            .expect("real C compiler required for this contract");
        assert!(version.status.success());
        let family = if String::from_utf8_lossy(&version.stdout)
            .to_ascii_lowercase()
            .contains("clang")
        {
            "clang"
        } else {
            "gcc"
        }
        .to_string();
        fs::write(root.join("config.h"), "#define OPTION 1\n#define VALUE 7\n").unwrap();
        let store = temp.path().join("store");
        Self {
            _temp: temp,
            root,
            store,
            compiler,
            family,
        }
    }
    fn context(&self, file: &str, compiler: Option<&str>, timeout: u64) {
        let value = serde_json::json!({"schema_version":"greppy.c-validation-context.v1","configuration":"option-one","compiler":{"family":self.family,"executable":compiler.unwrap_or(&self.compiler)},"translation_units":[{"file":file,"directory":".","arguments":["-std=c11","-Wall","-Wextra","-Werror","-fsyntax-only","{source}"]}],"input_files":["config.h"],"timeout_seconds":timeout});
        fs::write(
            self.root.join("context.json"),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap();
    }
    fn run(&self, args: &[&str], input: &[u8], selected: Option<&str>) -> Output {
        let mut command = Command::new(env!("CARGO_BIN_EXE_greppy"));
        command
            .current_dir(&self.root)
            .env("GREPPY_STORE_DIR", &self.store)
            .env("GREPPY_TEST_SKIP_INFERENCE", "1")
            .env_remove("GREPPY_VERIFY_TEST_COMMAND");
        if let Some(c) = selected {
            command.env("GREPPY_VERIFY_TEST_COMMAND", c);
        }
        let mut child = command
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(input).unwrap();
        child.wait_with_output().unwrap()
    }
}
fn value(out: &Output) -> serde_json::Value {
    serde_json::from_slice(&out.stdout).unwrap_or_else(|_| {
        panic!(
            "stdout={} stderr={}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    })
}
fn code(out: &Output, expected: i32) {
    assert_eq!(
        out.status.code(),
        Some(expected),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}
fn add_patch(file: &str, source: &str) -> String {
    format!(
        "*** Begin Patch\n*** Add File: {file}\n{}*** End Patch\n",
        source
            .lines()
            .map(|l| format!("+{l}\n"))
            .collect::<String>()
    )
}
#[test]
fn native_c_rejected_patch_never_publishes() {
    let f = Fixture::new();
    f.context("source.c", None, 20);
    let invalid = format!("{SOURCE}\nint broken(void) {{ return +; }}\n");
    let patch = add_patch("source.c", &invalid);
    let refused = f.run(
        &["patch", "--c-context", "context.json", "--json"],
        patch.as_bytes(),
        None,
    );
    code(&refused, 14);
    assert_eq!(
        value(&refused)["error"]["c_validation"]["status"],
        "rejected"
    );
    assert!(
        !f.root.join("source.c").exists(),
        "compiler-rejected transaction must not publish"
    );
}
#[test]
fn native_c_transaction_requires_real_compiler_success() {
    let f = Fixture::new();
    f.context("source.c", None, 20);
    for source in [SOURCE.to_string(), SOURCE.replace('\n', "\r\n")] {
        let default = f.run(&["write", "source.c", "--json"], source.as_bytes(), None);
        code(&default, 13);
        assert!(!f.root.join("source.c").exists());
        let accepted = f.run(
            &["write", "source.c", "--c-context", "context.json", "--json"],
            source.as_bytes(),
            None,
        );
        code(&accepted, 0);
        assert_eq!(
            fs::read(f.root.join("source.c")).unwrap(),
            source.as_bytes()
        );
        let record = value(&accepted);
        assert_eq!(record["c_validation"]["status"], "passed");
        assert_eq!(record["c_validation"]["coverage"], "selected-configuration");
        assert_eq!(record["c_validation"]["inactive_variants_checked"], false);
        assert!(
            record["c_validation"]["raw_parser_observations"][0]["errors"]
                .as_u64()
                .unwrap()
                > 0
        );
        fs::remove_file(f.root.join("source.c")).unwrap();
    }
    // A header and its consumer are checked together in the staged proposal.
    let patch=format!("*** Begin Patch\n*** Update File: config.h\n@@\n-#define VALUE 7\n+#define VALUE 9\n*** Add File: source.c\n{}*** End Patch\n",SOURCE.lines().map(|l|format!("+{l}\n")).collect::<String>());
    let accepted = f.run(
        &["patch", "--c-context", "context.json", "--json"],
        patch.as_bytes(),
        None,
    );
    code(&accepted, 0);
    assert!(fs::read_to_string(f.root.join("config.h"))
        .unwrap()
        .contains("VALUE 9"));
    assert_eq!(fs::read_to_string(f.root.join("source.c")).unwrap(), SOURCE);
}
#[test]
fn native_c_context_unavailable_and_stale_inputs_never_publish() {
    let f = Fixture::new();
    f.context("source.c", Some("/no/such/c/compiler"), 20);
    let out = f.run(
        &["write", "source.c", "--c-context", "context.json", "--json"],
        SOURCE.as_bytes(),
        None,
    );
    code(&out, 14);
    assert_eq!(
        value(&out)["error"]["c_validation"]["status"],
        "unavailable"
    );
    assert!(!f.root.join("source.c").exists());
    f.context("source.c", None, 20);
    let out = f.run(
        &["write", "source.c", "--c-context", "context.json", "--json"],
        SOURCE.replace("config.h", "missing.h").as_bytes(),
        None,
    );
    code(&out, 14);
    assert_eq!(
        value(&out)["error"]["c_validation"]["reason"],
        "missing_input"
    );
    assert!(!f.root.join("source.c").exists());
    f.context("unknown.c", None, 20);
    let out = f.run(
        &["write", "source.c", "--c-context", "context.json", "--json"],
        SOURCE.as_bytes(),
        None,
    );
    code(&out, 14);
    assert_eq!(
        value(&out)["error"]["c_validation"]["status"],
        "unavailable"
    );
    assert!(!f.root.join("source.c").exists());
    use std::os::unix::fs::PermissionsExt;
    let wrapper = f.root.join(format!(
        "compiler-{}.py",
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let script=format!("#!/usr/bin/python3\nimport os,sys,subprocess\nfrom pathlib import Path\ncompiler={}\nif '--version' in sys.argv: os.execv(compiler,[compiler,*sys.argv[1:]])\nresult=subprocess.run([compiler,*sys.argv[1:]])\nif '-fsyntax-only' in sys.argv and result.returncode==0: Path({}).write_text('#define OPTION 1\\n#define VALUE 11\\n')\nraise SystemExit(result.returncode)\n",serde_json::to_string(&f.compiler).unwrap(),serde_json::to_string(&f.root.join("config.h").display().to_string()).unwrap());
    fs::write(&wrapper, script).unwrap();
    fs::set_permissions(&wrapper, fs::Permissions::from_mode(0o700)).unwrap();
    f.context("source.c", Some(wrapper.to_str().unwrap()), 20);
    let out = f.run(
        &["write", "source.c", "--c-context", "context.json", "--json"],
        SOURCE.as_bytes(),
        None,
    );
    code(&out, 12);
    assert_eq!(value(&out)["error"]["c_validation"]["status"], "stale");
    assert!(!f.root.join("source.c").exists());
    let slow = f.root.join("slow-compiler.py");
    let script=format!("#!/usr/bin/python3\nimport os,sys,time\ncompiler={}\nif '--version' in sys.argv: os.execv(compiler,[compiler,*sys.argv[1:]])\ntime.sleep(10)\n",serde_json::to_string(&f.compiler).unwrap());
    fs::write(&slow, script).unwrap();
    fs::set_permissions(&slow, fs::Permissions::from_mode(0o700)).unwrap();
    f.context("source.c", Some(slow.to_str().unwrap()), 1);
    let out = f.run(
        &["write", "source.c", "--c-context", "context.json", "--json"],
        SOURCE.as_bytes(),
        None,
    );
    code(&out, 14);
    assert_eq!(value(&out)["error"]["c_validation"]["status"], "timed_out");
    assert!(!f.root.join("source.c").exists());
}
#[test]
fn native_c_mode_preserves_default_and_post_publication_verify() {
    let f = Fixture::new();
    f.context("source.c", None, 20);
    let original = b"int original(void) { return 7; }\n";
    fs::write(f.root.join("source.c"), original).unwrap();
    for option in [vec!["--c-context", "context.json"], vec!["--c-context=context.json"]] {
        let mut args = vec!["replace-lines", "source.c", "1:1", "int changed(void) { return 9; }"];
        args.extend(option);
        let out = f.run(&args, b"", None);
        code(&out, 64);
        assert!(String::from_utf8_lossy(&out.stdout).contains("supported only by write and patch"));
        assert_eq!(fs::read(f.root.join("source.c")).unwrap(), original);
    }
    let broken = b"int broken(void) { return undeclared_name; }\n";
    let selected = format!("{} -std=c11 -fsyntax-only source.c", f.compiler);
    let out = f.run(
        &["write", "source.c", "--verify", "--json"],
        broken,
        Some(&selected),
    );
    code(&out, 17);
    assert_eq!(value(&out)["published"], true);
    assert_eq!(fs::read(f.root.join("source.c")).unwrap(), broken);
}
