fn main() {
    println!("cargo:rerun-if-changed=platform/windows/winfsp_bridge.c");
    println!("cargo:rerun-if-env-changed=GREPPY_WINFSP_FORK_ROOT");
    if std::env::var_os("CARGO_CFG_TARGET_OS").as_deref() != Some(std::ffi::OsStr::new("windows")) {
        return;
    }
    let root = std::env::var_os("GREPPY_WINFSP_FORK_ROOT")
        .map(std::path::PathBuf::from)
        .expect("Windows provider builds require GREPPY_WINFSP_FORK_ROOT");
    let include = root.join("inc");
    let library = root.join("build/VStudio/build/Release");
    if !include.join("fuse3/fuse.h").is_file()
        || !library.join("greppyworkspacefsp-x64.lib").is_file()
    {
        panic!(
            "the exact Greppy WinFsp fork build is required under {}",
            root.display()
        );
    }
    cc::Build::new()
        .file("platform/windows/winfsp_bridge.c")
        .include(include.join("fuse3"))
        .include(&include)
        .define("_WIN32_WINNT", "0x0A00")
        .warnings(true)
        .compile("greppy_winfsp_bridge");
    println!("cargo:rustc-link-search=native={}", library.display());
    println!("cargo:rustc-link-lib=dylib=greppyworkspacefsp-x64");
    println!("cargo:rustc-link-lib=delayimp");
    println!(
        "cargo:rustc-link-arg-bin=greppy-workspace-provider=/DELAYLOAD:greppyworkspacefsp-x64.dll"
    );
}
