fn main() {
    println!("cargo:rerun-if-changed=platform/windows/winfsp_bridge.c");
    if std::env::var_os("CARGO_CFG_TARGET_OS").as_deref() != Some(std::ffi::OsStr::new("windows")) {
        return;
    }
    let program_files = std::env::var_os("ProgramFiles(x86)")
        .or_else(|| std::env::var_os("ProgramFiles"))
        .expect("WinFsp build requires Program Files");
    let root = std::path::PathBuf::from(program_files).join("WinFsp");
    let include = root.join("inc");
    let library = root.join("lib");
    if !include.join("fuse3/fuse.h").is_file() || !library.join("winfsp-x64.lib").is_file() {
        panic!(
            "WinFsp 2.1 Developer files are required under {}",
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
    println!("cargo:rustc-link-lib=dylib=winfsp-x64");
}
