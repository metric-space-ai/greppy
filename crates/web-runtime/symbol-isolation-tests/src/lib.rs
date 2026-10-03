#[allow(dead_code)]
#[path = "../../build-support/localize_js_static.rs"]
mod localize_js_static;

#[cfg(all(test, target_os = "linux", target_arch = "x86_64"))]
mod tests {
    use super::localize_js_static::{
        local_icu_comdat_signatures, redefine_linux_archive_symbols, verify_local_icu_comdat_absent,
    };
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::Path;
    use std::process::Command;

    const ICU_COMDAT: &str = "_ZN6icu_7715MaybeStackArrayIcLi40EEC5Ev";

    fn run(program: &str, args: &[&str], cwd: &Path) {
        let output = Command::new(program)
            .args(args)
            .current_dir(cwd)
            .output()
            .unwrap_or_else(|error| panic!("run {program}: {error}"));
        assert!(
            output.status.success(),
            "{program} failed: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn group_assembly(function: &str) -> String {
        format!(
            ".section .text.{function},\"axG\",@progbits,{ICU_COMDAT},comdat\n\
             .globl {function}\n\
             .type {function},@function\n\
             {function}:\n\
             ret\n"
        )
    }

    #[test]
    fn namespaces_icu_comdat_signatures_and_preserves_both_engines_at_link() {
        let temp =
            std::env::temp_dir().join(format!("greppy-comdat-isolation-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(&temp).unwrap();
        fs::write(temp.join("sm.s"), group_assembly("sm_ctor")).unwrap();
        fs::write(temp.join("v8.s"), group_assembly("v8_ctor")).unwrap();
        fs::write(
            temp.join("main.c"),
            "void sm_ctor(void); void v8_ctor(void); int main(void) { sm_ctor(); v8_ctor(); return 0; }\n",
        )
        .unwrap();
        run("cc", &["-c", "sm.s", "-o", "sm.o"], &temp);
        run("cc", &["-c", "v8.s", "-o", "v8.o"], &temp);
        run("ar", &["crs", "libsm.a", "sm.o"], &temp);
        run("ar", &["crs", "libv8.a", "v8.o"], &temp);

        let sm = temp.join("libsm.a");
        let signatures = local_icu_comdat_signatures(&sm, "__greppy_sm_").unwrap();
        assert_eq!(signatures, BTreeSet::from([ICU_COMDAT.to_owned()]));
        redefine_linux_archive_symbols(&sm, &signatures, "__greppy_sm_").unwrap();
        verify_local_icu_comdat_absent(&sm, "__greppy_sm_").unwrap();

        // Re-running discovery and repair must be a no-op for a cached archive.
        let pending = local_icu_comdat_signatures(&sm, "__greppy_sm_").unwrap();
        assert!(pending.is_empty());
        redefine_linux_archive_symbols(&sm, &pending, "__greppy_sm_").unwrap();

        run(
            "cc",
            &["main.c", "libsm.a", "libv8.a", "-o", "comdat-link"],
            &temp,
        );
        run("./comdat-link", &[], &temp);
        let _ = fs::remove_dir_all(temp);
    }
}
