include!("../build-support/localize_js_static.rs");

fn main() {
    println!("cargo:rerun-if-changed=../build-support/localize_js_static.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CONTROLLER_RUNTIME");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CONTENT_RUNTIME");

    // The symbol collision exists only in the combined production runtime:
    // controller-runtime brings V8 through deno_core, while content-runtime
    // brings SpiderMonkey through Servo. Focused controller/sandbox builds use
    // `--no-default-features` and intentionally contain neither engine; making
    // those gates discover libjs_static.a contradicted their feature contract
    // and made the Linux Landlock CI fail before running a single test.
    let controller = std::env::var_os("CARGO_FEATURE_CONTROLLER_RUNTIME").is_some();
    let content = std::env::var_os("CARGO_FEATURE_CONTENT_RUNTIME").is_some();
    if controller && content {
        if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
            install_linux_linker_adapter();
        } else {
            localize_mozjs_js_static();
        }
    }
}

fn install_linux_linker_adapter() {
    println!("cargo:rerun-if-changed=../build-support/late_linker.rs");
    println!("cargo:rerun-if-env-changed=RUSTC_LINKER");
    if std::env::var("HOST") != std::env::var("TARGET") {
        panic!("combined Linux runtime currently requires a native GNU linker driver");
    }
    let driver = std::env::var_os("RUSTC_LINKER").unwrap_or_else(|| "cc".into());
    let version = std::process::Command::new(&driver).arg("--version").output()
        .expect("inspect the configured linker driver");
    let text = String::from_utf8_lossy(&version.stdout);
    if !version.status.success() || !text.contains("Free Software Foundation")
        || text.to_ascii_lowercase().contains("clang")
    {
        panic!("combined Linux runtime link isolation requires a GNU compiler driver; configured driver: {:?}",driver);
    }
    let output = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo OUT_DIR"));
    let adapter = output.join("greppy-engine-linker");
    let source = std::path::Path::new("../build-support/late_linker.rs");
    let compiler = std::env::var_os("RUSTC").expect("Cargo RUSTC");
    let result = std::process::Command::new(compiler)
        .args(["--edition=2021","--crate-name","greppy_engine_linker","-O",
               "-C","codegen-units=1","-C","link-arg=-Wl,--threads=2"])
        .arg(source).arg("-o").arg(&adapter).status().expect("compile private linker adapter");
    assert!(result.success(),"compile private linker adapter failed");
    let path = adapter.to_str().expect("linker adapter path must be UTF-8");
    assert!(!path.chars().any(|c| matches!(c, ',' | '\n' | '\r')),"GCC wrapper path cannot contain comma or newline");
    println!("cargo:rustc-link-arg=-wrapper");
    println!("cargo:rustc-link-arg={path}");
}

