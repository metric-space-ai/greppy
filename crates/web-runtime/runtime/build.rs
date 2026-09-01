include!("../build-support/localize_js_static.rs");

fn main() {
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
        localize_mozjs_js_static();
    }
}
