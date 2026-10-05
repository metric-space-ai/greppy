use std::io::Write;
use std::process::{Command, Stdio};
struct OwnedChild(Option<std::process::Child>);
impl Drop for OwnedChild {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
fn finish(mut child: OwnedChild) -> std::process::Output {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    loop {
        if child.0.as_mut().unwrap().try_wait().unwrap().is_some() {
            return child.0.take().unwrap().wait_with_output().unwrap();
        }
        assert!(
            std::time::Instant::now() < deadline,
            "owned macro validation CLI exceeded15s"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}
fn write_source(
    repo: &std::path::Path,
    store: &std::path::Path,
    path: &str,
    source: &[u8],
) -> std::process::Output {
    let mut child = OwnedChild(Some(
        Command::new(env!("CARGO_BIN_EXE_greppy"))
            .current_dir(repo)
            .env("GREPPY_STORE_DIR", store)
            .env("GREPPY_TEST_SKIP_INFERENCE", "1")
            .args(["--json", "write", path])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    ));
    child
        .0
        .as_mut()
        .unwrap()
        .stdin
        .take()
        .unwrap()
        .write_all(source)
        .unwrap();
    finish(child)
}
#[test]
fn local_jni_and_xmacro_writes_preserve_exact_bytes() {
    let temp = tempfile::tempdir().unwrap();
    let repo = temp.path().join("repo");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    let store = temp.path().join("store");
    for (path, source) in [
        ("jni.c", "#define JNIEXPORT __attribute__((visibility(\"default\")))\n#define JNICALL\nJNIEXPORT int JNICALL probe(void) { return 0; }\n"),
        ("fields.c", "#define FIELDS(X) X(int, count) X(float, ratio)\n#define DECL(type,name) type name;\nstruct record { FIELDS(DECL) };\n"),
    ] {
        let output = write_source(&repo, &store, path, source.as_bytes());
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stdout));
        assert_eq!(std::fs::read(repo.join(path)).unwrap(), source.as_bytes());
    }
    let original = std::fs::read(repo.join("jni.c")).unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_greppy"))
        .current_dir(&repo)
        .env("GREPPY_STORE_DIR", &store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .args(["replace-text", "jni.c", "return 0;", "return 0"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let output = finish(OwnedChild(Some(child)));
    assert_eq!(output.status.code(), Some(13));
    assert_eq!(std::fs::read(repo.join("jni.c")).unwrap(), original);
}
#[test]
fn unsupported_macro_write_has_original_invocation_diagnostic() {
    let temp = tempfile::tempdir().unwrap();
    let repo = temp.path().join("repo");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    let source = b"#define JOIN(a,b) a ## b\nint JOIN(a,b);\n";
    let output = write_source(&repo, &temp.path().join("store"), "unsupported.c", source);
    assert_eq!(output.status.code(), Some(13));
    let record: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(record["published"], false);
    assert!(record["error"]["message"]
        .as_str()
        .unwrap()
        .contains("unsupported.c:2:5"));
    assert!(!repo.join("unsupported.c").exists());
}
