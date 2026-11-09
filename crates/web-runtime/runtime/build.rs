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
    let machine = std::process::Command::new(&driver).arg("-dumpmachine").output()
        .expect("inspect GNU compiler target");
    let machine_name = String::from_utf8_lossy(&machine.stdout);
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").expect("Cargo target architecture");
    let native_machine = machine_name.trim().split('-').next() == Some(arch.as_str());
    if !version.status.success() || !text.contains("Free Software Foundation")
        || text.to_ascii_lowercase().contains("clang")
        || !machine.status.success() || !native_machine
    {
        panic!("combined Linux runtime link isolation requires a native GNU compiler driver; configured driver: {:?}",driver);
    }
    let bfd = std::process::Command::new(&driver).arg("-print-prog-name=ld.bfd").output()
        .expect("locate the GNU driver's BFD linker");
    assert!(bfd.status.success(),"GNU driver could not locate ld.bfd");
    let bfd_path = String::from_utf8(bfd.stdout).expect("GNU BFD path must be UTF-8");
    let bfd_version = std::process::Command::new(bfd_path.trim()).arg("--version").output()
        .expect("inspect GNU BFD linker");
    assert!(bfd_version.status.success()
        && String::from_utf8_lossy(&bfd_version.stdout).starts_with("GNU ld"),
        "the helper requires the verified GNU BFD linker");
    let output = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo OUT_DIR"));
    let adapter = output.join("greppy-engine-linker");
    let source = std::path::Path::new("../build-support/late_linker.rs");
    let compiler = std::env::var_os("RUSTC").expect("Cargo RUSTC");
    let mut linker_option = std::ffi::OsString::from("linker=");
    linker_option.push(&driver);
    // Bind the helper to the verified driver and single-threaded GNU BFD.
    // Do not assume the final target's LLD-only --threads flag applies here.
    let result = std::process::Command::new(compiler)
        .args(["--edition=2021","--crate-name","greppy_engine_linker","-O",
               "-C","codegen-units=1","-C","link-arg=-fuse-ld=bfd","-C"])
        .arg(linker_option).arg(source).arg("-o").arg(&adapter)
        .status().expect("compile private linker adapter");
    assert!(result.success(),"compile private linker adapter failed");
    let path = adapter.to_str().expect("linker adapter path must be UTF-8");
    assert!(!path.chars().any(|c| matches!(c, ',' | '\n' | '\r')),"GCC wrapper path cannot contain comma or newline");
    println!("cargo:rustc-link-arg=-wrapper");
    println!("cargo:rustc-link-arg={path}");
}

