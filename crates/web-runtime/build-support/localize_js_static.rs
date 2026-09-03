use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const UNIX_IRREGEXP_MEMBER: &str = "Unified_cpp_js_src_irregexp0.o";
const WINDOWS_IRREGEXP_MEMBER: &str = "Unified_cpp_js_src_irregexp0.obj";

/// V8/SpiderMonkey irregexp overlap: ld64 coalesces these and V8 Isolate::Delete
/// runs the SpiderMonkey destructor (KERN_INVALID_ADDRESS).
///
/// `nmedit -R` only turns SM copies into "non-external (was a private external)".
/// ld64 still coalesces those with V8's private-extern destructors. Proven crash:
/// `Isolate::Delete` → `Deinitialize` → SM `Isolate::~Isolate` (`mozilla::SegmentedVector`).
/// Rename the SM symbol *strings* (same length) so the final bind cannot pick them.
/// Intra-object relocations follow the symbol index, not the string, so SM still
/// calls its own destructor. Keep js::irregexp::* exported.
const RENAME_SYMBOLS: &[(&str, &str)] = &[
    (
        "__ZN2v88internal7IsolateD1Ev",
        "__ZN2sm8internal7IsolateD1Ev",
    ),
    (
        "__ZN2v88internal7IsolateD2Ev",
        "__ZN2sm8internal7IsolateD2Ev",
    ),
    (
        "__ZN2v88internal6PrintFEPKcz",
        "__ZN2sm8internal6PrintFEPKcz",
    ),
    (
        "__ZN2v88internal6PrintFEP7__sFILEPKcz",
        "__ZN2sm8internal6PrintFEP7__sFILEPKcz",
    ),
];

// ELF uses one leading underscore for Itanium C++ names, and glibc spells
// FILE as `_IO_FILE`. Keep this list bound to the measured Linux archive
// intersection; the Mach-O names above are not portable objcopy inputs.
const LINUX_RENAME_SYMBOLS: &[(&str, &str)] = &[
    ("_ZN2v88internal7IsolateD1Ev", "_ZN2sm8internal7IsolateD1Ev"),
    ("_ZN2v88internal7IsolateD2Ev", "_ZN2sm8internal7IsolateD2Ev"),
    ("_ZN2v88internal6PrintFEPKcz", "_ZN2sm8internal6PrintFEPKcz"),
    (
        "_ZN2v88internal6PrintFEP8_IO_FILEPKcz",
        "_ZN2sm8internal6PrintFEP8_IO_FILEPKcz",
    ),
];

// The same two PrintF definitions overlap under the MSVC ABI. The Isolate
// destructor does not: SpiderMonkey exports the public QEAA form while V8's
// copy is the private AEAA form. Keep this list tied to the measured COFF
// symbol intersection instead of transliterating the Itanium list.
const WINDOWS_RENAME_SYMBOLS: &[(&str, &str)] = &[
    (
        "?PrintF@internal@v8@@YAXPEBDZZ",
        "?GreppySMPrintF@internal@v8@@YAXPEBDZZ",
    ),
    (
        "?PrintF@internal@v8@@YAXPEAU_iobuf@@PEBDZZ",
        "?GreppySMPrintF@internal@v8@@YAXPEAU_iobuf@@PEBDZZ",
    ),
];

pub fn localize_mozjs_js_static() {
    let archives = find_js_static_archives();
    if archives.is_empty() {
        panic!("mozjs_sys js_static archive not found; cannot localize V8/SpiderMonkey overlap");
    }
    for archive in &archives {
        if let Err(error) = localize_archive(archive) {
            panic!("localize {}: {error}", archive.display());
        }
        if let Err(error) = verify_hidden(archive) {
            panic!("verify {}: {error}", archive.display());
        }
        println!("cargo:rerun-if-changed={}", archive.display());
    }
    let rlibs = find_mozjs_sys_rlibs();
    if rlibs.is_empty() {
        panic!("libmozjs_sys-*.rlib not found; irregexp objects are linked from the rlib, not only libjs_static.a");
    }
    for rlib in &rlibs {
        if let Err(error) = localize_archive(rlib) {
            panic!("localize rlib {}: {error}", rlib.display());
        }
        println!("cargo:rerun-if-changed={}", rlib.display());
    }
    let Some(rusty_v8) = find_rusty_v8() else {
        panic!("librusty_v8.a not found; cannot run defined-symbol intersection gate");
    };
    let mut v8_symbol_sources = vec![rusty_v8];
    if cfg!(target_os = "linux") {
        let v8_rlibs = find_rlibs("libv8-");
        if v8_rlibs.is_empty() {
            panic!(
                "libv8-*.rlib not found; the final Linux link takes bundled ICU symbols from the rlib"
            );
        }
        v8_symbol_sources.extend(v8_rlibs);
    }
    for source in &v8_symbol_sources {
        println!("cargo:rerun-if-changed={}", source.display());
    }
    if cfg!(target_os = "linux") {
        if let Err(error) =
            namespace_linux_engine_symbols(&archives, &rlibs, &v8_symbol_sources)
        {
            panic!("namespace Linux SpiderMonkey symbols: {error}");
        }
    }
    for archive in &archives {
        for v8_source in &v8_symbol_sources {
            if let Err(error) = assert_symbol_intersection(archive, v8_source) {
                panic!(
                    "symbol intersection {} versus {}: {error}",
                    archive.display(),
                    v8_source.display()
                );
            }
        }
    }
}

fn find_mozjs_sys_rlibs() -> Vec<PathBuf> {
    find_rlibs("libmozjs_sys-")
}

fn find_rlibs(prefix: &str) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for root in search_roots() {
        let deps = root.join("deps");
        let Ok(entries) = fs::read_dir(&deps) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(prefix) && name.ends_with(".rlib") {
                let path = entry.path();
                if path.is_file() && !found.contains(&path) {
                    found.push(path);
                }
            }
        }
    }
    found
}

fn namespace_linux_engine_symbols(
    js_static_archives: &[PathBuf],
    mozjs_rlibs: &[PathBuf],
    v8_symbol_sources: &[PathBuf],
) -> Result<(), String> {
    let mut v8_globals = BTreeSet::new();
    for source in v8_symbol_sources {
        v8_globals.extend(defined_globals(source)?);
    }

    // Both engines embed ICU 77. ELF does not coalesce those strong C/C++
    // definitions as ld64 does, so merely calling the overlap "permitted"
    // still makes the final link fail. Rename the SpiderMonkey copy, including
    // every reference in each archive, and leave V8's public symbol names as
    // the process-wide owner.
    for archive in js_static_archives.iter().chain(mozjs_rlibs) {
        let globals = defined_globals(archive)?;
        let overlaps = globals
            .intersection(&v8_globals)
            .filter(|name| is_icu(name))
            .cloned()
            .collect::<BTreeSet<_>>();
        reject_mixed_icu_versions(&overlaps)?;
        redefine_linux_archive_symbols(archive, &overlaps, "__greppy_sm_")?;
        verify_symbols_absent(archive, &overlaps, "ICU")?;
    }

    // mozjs_sys uses diplomat-runtime 0.8 through icu_capi while V8's
    // temporal bindings use diplomat-runtime 0.16. Both crates export the
    // same unmangled allocator entry points from otherwise incompatible
    // versions. Namespace the older side and all of its possible callers.
    let diplomat_symbols =
        BTreeSet::from(["diplomat_alloc".to_owned(), "diplomat_free".to_owned()]);
    let diplomat_rlibs = find_rlibs("libdiplomat_runtime-");
    let old_diplomat = diplomat_rlibs
        .iter()
        .filter_map(|archive| {
            defined_globals(archive)
                .ok()
                .filter(|symbols| symbols.contains("diplomat_buffer_write_create"))
                .map(|_| archive.clone())
        })
        .collect::<Vec<_>>();
    if old_diplomat.is_empty() {
        return Err("diplomat-runtime 0.8 archive not found by its ABI marker".into());
    }
    let icu_capi_rlibs = find_rlibs("libicu_capi-");
    for archive in old_diplomat
        .iter()
        .chain(icu_capi_rlibs.iter())
        .chain(mozjs_rlibs.iter())
    {
        redefine_linux_archive_symbols(archive, &diplomat_symbols, "greppy_sm_")?;
    }
    for archive in &old_diplomat {
        verify_symbols_absent(archive, &diplomat_symbols, "diplomat-runtime")?;
    }
    Ok(())
}

fn reject_mixed_icu_versions(symbols: &BTreeSet<String>) -> Result<(), String> {
    if let Some(name) = symbols
        .iter()
        .find(|name| name.contains("icu_76") || name.contains("icudt76") || name.contains("_76"))
    {
        return Err(format!(
            "ICU overlap {name} is not ICU 77; refusing mixed ICU versions"
        ));
    }
    Ok(())
}

fn redefine_linux_archive_symbols(
    archive: &Path,
    symbols: &BTreeSet<String>,
    prefix: &str,
) -> Result<(), String> {
    if symbols.is_empty() {
        return Ok(());
    }
    let parent = archive
        .parent()
        .ok_or_else(|| format!("archive has no parent: {}", archive.display()))?;
    let map = parent.join(format!(
        ".greppy-redefine-{}-{}.txt",
        std::process::id(),
        archive.file_name().unwrap_or_default().to_string_lossy()
    ));
    let contents = symbols
        .iter()
        .map(|name| format!("{name} {prefix}{name}\n"))
        .collect::<String>();
    fs::write(&map, contents).map_err(|e| format!("write {}: {e}", map.display()))?;
    let output = Command::new("objcopy")
        .arg(format!("--redefine-syms={}", map.display()))
        .arg(archive)
        .output()
        .map_err(|e| format!("objcopy redefine {}: {e}", archive.display()));
    let _ = fs::remove_file(&map);
    let output = output?;
    if !output.status.success() {
        return Err(format!(
            "objcopy redefine {} failed: {}\n{}",
            archive.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(())
}

fn verify_symbols_absent(
    archive: &Path,
    symbols: &BTreeSet<String>,
    family: &str,
) -> Result<(), String> {
    let remaining = defined_globals(archive)?
        .intersection(symbols)
        .cloned()
        .collect::<Vec<_>>();
    if remaining.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{family} symbols remain globally defined in {}: {:?}",
        archive.display(),
        remaining.iter().take(20).collect::<Vec<_>>()
    ))
}

fn find_js_static_archives() -> Vec<PathBuf> {
    let archive_name = if cfg!(windows) {
        "js_static.lib"
    } else {
        "libjs_static.a"
    };
    let mut found = Vec::new();
    for root in build_roots() {
        let Ok(entries) = fs::read_dir(&root) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            if !name.to_string_lossy().starts_with("mozjs_sys-") {
                continue;
            }
            let archive = entry
                .path()
                .join("out")
                .join("build")
                .join("js")
                .join("src")
                .join("build")
                .join(archive_name);
            if archive.is_file() && !found.contains(&archive) {
                found.push(archive);
            }
        }
    }
    found
}

fn find_rusty_v8() -> Option<PathBuf> {
    let archive_name = if cfg!(windows) {
        "rusty_v8.lib"
    } else {
        "librusty_v8.a"
    };
    for root in search_roots() {
        let candidate = root.join("gn_out").join("obj").join(archive_name);
        if candidate.is_file() {
            return Some(candidate);
        }
        if let Ok(entries) = fs::read_dir(root.join("build")) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                if !name.to_string_lossy().starts_with("rusty_v8-") {
                    continue;
                }
                let nested = entry.path().join("out").join("obj").join(archive_name);
                if nested.is_file() {
                    return Some(nested);
                }
            }
        }
    }
    None
}

fn search_roots() -> Vec<PathBuf> {
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".into());
    let mut roots = Vec::new();
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
    if let Some(workspace) = manifest.parent() {
        roots.push(workspace.join("target").join(&profile));
        roots.push(
            workspace
                .join("target")
                .join(env::var("TARGET").unwrap_or_default())
                .join(&profile),
        );
    }
    if let Ok(dir) = env::var("CARGO_TARGET_DIR") {
        let dir = PathBuf::from(dir);
        roots.push(dir.join(&profile));
        roots.push(
            dir.join(env::var("TARGET").unwrap_or_default())
                .join(&profile),
        );
    }
    if let Ok(out) = env::var("OUT_DIR") {
        let out = PathBuf::from(out);
        if let Some(profile_dir) = out.parent().and_then(Path::parent).and_then(Path::parent) {
            roots.push(profile_dir.to_path_buf());
        }
    }
    roots
}

fn build_roots() -> Vec<PathBuf> {
    search_roots()
        .into_iter()
        .map(|root| root.join("build"))
        .collect()
}

fn localize_archive(archive: &Path) -> Result<(), String> {
    let parent = archive
        .parent()
        .ok_or_else(|| format!("archive has no parent: {}", archive.display()))?;
    let work = parent.join("greppy-localize-irregexp");
    let _ = fs::remove_dir_all(&work);
    fs::create_dir_all(&work).map_err(|e| format!("mkdir {}: {e}", work.display()))?;
    let member_name = irregexp_member_name(archive)?;
    let status = Command::new(archive_tool())
        .args(["x", archive.to_str().unwrap(), &member_name])
        .current_dir(&work)
        .status()
        .map_err(|e| format!("{} x: {e}", archive_tool()))?;
    if !status.success() {
        return Err(format!("{} x failed: {status}", archive_tool()));
    }
    let member_basename = Path::new(&member_name)
        .file_name()
        .ok_or_else(|| format!("archive member has no file name: {member_name}"))?;
    let member = work.join(member_basename);
    if !member.is_file() {
        return Err(format!("missing {member_name} in {}", archive.display()));
    }
    rename_v8_overlap_symbols(&member)?;
    let archive_str = archive.to_str().unwrap();
    let _ = Command::new(archive_tool())
        .args(["d", archive_str, &member_name])
        .status();
    let status = Command::new(archive_tool())
        .args(["r", archive_str, member_basename.to_str().unwrap()])
        .current_dir(&work)
        .status()
        .map_err(|e| format!("{} r: {e}", archive_tool()))?;
    if !status.success() {
        return Err(format!("{} r failed: {status}", archive_tool()));
    }
    if !cfg!(windows) && archive.extension().and_then(|e| e.to_str()) != Some("rlib") {
        let _ = Command::new("ranlib").arg(archive).status();
    }
    Ok(())
}

fn archive_tool() -> &'static str {
    if cfg!(windows) {
        "llvm-ar"
    } else {
        "ar"
    }
}

fn irregexp_member_name(archive: &Path) -> Result<String, String> {
    let wanted = if cfg!(windows) {
        WINDOWS_IRREGEXP_MEMBER
    } else {
        UNIX_IRREGEXP_MEMBER
    };
    let output = Command::new(archive_tool())
        .args(["t", archive.to_str().unwrap()])
        .output()
        .map_err(|e| format!("{} t {}: {e}", archive_tool(), archive.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} t {} failed: {}\n{}",
            archive_tool(),
            archive.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .find(|name| {
            name.replace('\\', "/")
                .rsplit('/')
                .next()
                .is_some_and(|base| base == wanted)
        })
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("missing {wanted} in {}", archive.display()))
}

fn verify_hidden(archive: &Path) -> Result<(), String> {
    let defined = defined_symbols(archive)?;
    let hidden = if cfg!(windows) {
        WINDOWS_RENAME_SYMBOLS
    } else if cfg!(target_os = "linux") {
        LINUX_RENAME_SYMBOLS
    } else {
        RENAME_SYMBOLS
    };
    let still: Vec<_> = hidden
        .iter()
        .map(|(from, _)| *from)
        .filter(|name| defined.contains(*name))
        .collect();
    if !still.is_empty() {
        return Err(format!(
            "SpiderMonkey still defines V8-overlapping symbols (local or global): {still:?}"
        ));
    }
    Ok(())
}

fn assert_symbol_intersection(js_static: &Path, rusty_v8: &Path) -> Result<(), String> {
    let sm = defined_globals(js_static)?;
    let v8 = defined_globals(rusty_v8)?;
    let mut permitted = 0usize;
    let mut dangerous = Vec::new();
    for name in sm.intersection(&v8) {
        if is_permitted_overlap(name) {
            permitted += 1;
            continue;
        }
        dangerous.push(name.clone());
    }
    if !dangerous.is_empty() {
        let preview: Vec<_> = dangerous.iter().take(20).cloned().collect();
        return Err(format!(
            "{} non-ICU overlapping defined globals between libjs_static.a and librusty_v8.a: {preview:?}",
            dangerous.len()
        ));
    }
    println!("cargo:warning=engine symbol intersection: 0 dangerous overlaps, {permitted} permitted same-toolchain ICU/CRT overlaps");
    Ok(())
}

fn is_icu(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("icu")
        || name.contains("_77")
        || name.contains("UCaseMap")
        || name.contains("CollatorSpec")
        || name.contains("UErrorCode")
        || name.contains("CReg")
        || name.starts_with("uprv_")
}

fn is_permitted_overlap(name: &str) -> bool {
    (cfg!(target_os = "macos") && is_icu(name))
        || (cfg!(windows) && is_permitted_windows_crt_overlap(name))
}

fn is_permitted_windows_crt_overlap(name: &str) -> bool {
    matches!(
        name,
        "?_OptionsStorage@?1??__local_stdio_printf_options@@9@4_KA"
            | "?what@bad_optional_access@std@@UEBAPEBDXZ"
            | "?abort_noreturn@@YAXXZ"
            | "_Avx2WmemEnabledWeakValue"
            | "__local_stdio_printf_options"
            | "fprintf"
            | "printf"
            | "snprintf"
    )
}

fn defined_globals(archive: &Path) -> Result<BTreeSet<String>, String> {
    let extra: &[&str] = if cfg!(target_os = "macos") {
        &["-gU"]
    } else {
        &["-g", "--defined-only"]
    };
    defined_symbols_with(archive, extra)
}

fn defined_symbols(archive: &Path) -> Result<BTreeSet<String>, String> {
    defined_symbols_with(archive, &[])
}

fn defined_symbols_with(archive: &Path, extra: &[&str]) -> Result<BTreeSet<String>, String> {
    let mut args: Vec<&str> = extra.to_vec();
    let path = archive.to_str().unwrap();
    args.push(path);
    let nm = if cfg!(windows) { "llvm-nm" } else { "nm" };
    let output = Command::new(nm)
        .args(&args)
        .output()
        .map_err(|e| format!("{nm} {}: {e}", archive.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{nm} {} failed: {}\n{}",
            archive.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let mut names = BTreeSet::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let mut parts = line.split_whitespace();
        let Some(kind) = parts.nth(1) else {
            continue;
        };
        if kind != "T"
            && kind != "t"
            && kind != "S"
            && kind != "s"
            && kind != "D"
            && kind != "d"
            && kind != "B"
            && kind != "b"
            && kind != "C"
        {
            continue;
        }
        if let Some(name) = parts.next() {
            names.insert(name.to_owned());
        }
    }
    Ok(names)
}

fn rename_v8_overlap_symbols(object: &Path) -> Result<(), String> {
    let renames = if cfg!(windows) {
        WINDOWS_RENAME_SYMBOLS
    } else if cfg!(target_os = "linux") {
        LINUX_RENAME_SYMBOLS
    } else {
        RENAME_SYMBOLS
    };
    for (from, to) in renames {
        if cfg!(target_os = "macos") && from.len() != to.len() {
            return Err(format!("rename length mismatch {from} -> {to}"));
        }
    }
    if cfg!(windows) || cfg!(target_os = "linux") {
        let objcopy = if cfg!(windows) {
            "llvm-objcopy"
        } else {
            "objcopy"
        };
        let mut cmd = Command::new(objcopy);
        for (from, to) in renames {
            cmd.arg(format!("--redefine-sym={from}={to}"));
        }
        let output = cmd
            .arg(object)
            .output()
            .map_err(|e| format!("{objcopy} redefine: {e}"))?;
        if !output.status.success() {
            return Err(format!(
                "{objcopy} redefine failed: {}\n{}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        return Ok(());
    }
    for (from, to) in renames {
        if from.len() != to.len() {
            return Err(format!("rename length mismatch {from} -> {to}"));
        }
    }
    let mut bytes = fs::read(object).map_err(|e| format!("read {}: {e}", object.display()))?;
    for (from, to) in renames {
        let needle = format!("{from}\0").into_bytes();
        let replacement = format!("{to}\0").into_bytes();
        let mut replaced = 0usize;
        let mut i = 0usize;
        while let Some(pos) = find_bytes(&bytes[i..], &needle) {
            let at = i + pos;
            bytes[at..at + replacement.len()].copy_from_slice(&replacement);
            replaced += 1;
            i = at + replacement.len();
        }
        if replaced == 0 && !object_defines(&bytes, to) {
            return Err(format!(
                "symbol {from} not found as a NUL-terminated string in {}",
                object.display()
            ));
        }
    }
    clear_pext_on_renamed_symbols(&mut bytes)?;
    fs::write(object, bytes).map_err(|e| format!("write {}: {e}", object.display()))?;
    Ok(())
}

fn object_defines(bytes: &[u8], name: &str) -> bool {
    find_bytes(bytes, format!("{name}\0").as_bytes()).is_some()
}

/// nmedit -R leaves N_PEXT set ("non-external (was a private external)").
/// ld64 still coalesces those with V8 private-extern Isolate destructors.
/// Clear N_PEXT|N_EXT so the SM copy is a classic local (N_SECT only).
fn clear_pext_on_renamed_symbols(bytes: &mut [u8]) -> Result<(), String> {
    const MH_MAGIC_64: u32 = 0xFEED_FACF;
    const LC_SYMTAB: u32 = 2;
    const N_EXT: u8 = 0x01;
    const N_PEXT: u8 = 0x10;
    const N_SECT: u8 = 0x0e;
    if bytes.len() < 32 {
        return Err("object too small to be Mach-O".into());
    }
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    if magic != MH_MAGIC_64 {
        return Ok(());
    }
    let ncmds = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
    let mut off = 32usize;
    let mut stroff = 0u32;
    let mut nsyms = 0u32;
    let mut symoff = 0u32;
    for _ in 0..ncmds {
        if off + 8 > bytes.len() {
            return Err("truncated load commands".into());
        }
        let cmd = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let cmdsize = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap()) as usize;
        if cmd == LC_SYMTAB && cmdsize >= 24 {
            symoff = u32::from_le_bytes(bytes[off + 8..off + 12].try_into().unwrap());
            nsyms = u32::from_le_bytes(bytes[off + 12..off + 16].try_into().unwrap());
            stroff = u32::from_le_bytes(bytes[off + 16..off + 20].try_into().unwrap());
        }
        off += cmdsize;
    }
    let interesting: Vec<&str> = RENAME_SYMBOLS
        .iter()
        .flat_map(|(from, to)| [*from, *to])
        .collect();
    let mut patched = 0u32;
    for i in 0..nsyms {
        let eoff = symoff as usize + i as usize * 16;
        if eoff + 16 > bytes.len() {
            return Err("truncated nlist".into());
        }
        let n_strx = u32::from_le_bytes(bytes[eoff..eoff + 4].try_into().unwrap());
        let sstart = stroff as usize + n_strx as usize;
        if sstart >= bytes.len() {
            continue;
        }
        let send = bytes[sstart..]
            .iter()
            .position(|&b| b == 0)
            .map(|p| sstart + p)
            .unwrap_or(bytes.len());
        let Ok(name) = std::str::from_utf8(&bytes[sstart..send]) else {
            continue;
        };
        if !interesting.contains(&name) {
            continue;
        }
        bytes[eoff + 4] = N_SECT;
        let _ = (N_EXT, N_PEXT);
        patched += 1;
    }
    if patched == 0 {
        return Err("no overlapping Isolate/PrintF nlist entries to clear N_PEXT on".into());
    }
    Ok(())
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::{
        is_permitted_overlap, is_permitted_windows_crt_overlap, reject_mixed_icu_versions,
        LINUX_RENAME_SYMBOLS,
    };
    use std::collections::BTreeSet;

    #[test]
    fn linux_irregexp_renames_match_the_measured_elf_overlap() {
        let sources = LINUX_RENAME_SYMBOLS
            .iter()
            .map(|(from, _)| *from)
            .collect::<Vec<_>>();
        assert_eq!(
            sources,
            vec![
                "_ZN2v88internal7IsolateD1Ev",
                "_ZN2v88internal7IsolateD2Ev",
                "_ZN2v88internal6PrintFEPKcz",
                "_ZN2v88internal6PrintFEP8_IO_FILEPKcz",
            ]
        );
        assert!(LINUX_RENAME_SYMBOLS
            .iter()
            .all(|(from, to)| from.len() == to.len()));
    }

    #[test]
    fn windows_bad_optional_access_comdat_is_permitted_but_v8_shims_are_not() {
        assert!(is_permitted_windows_crt_overlap(
            "?what@bad_optional_access@std@@UEBAPEBDXZ"
        ));
        assert!(!is_permitted_windows_crt_overlap(
            "?PrintF@internal@v8@@YAXPEBDZZ"
        ));
    }

    #[test]
    fn linux_never_treats_icu_overlap_as_linker_safe() {
        if cfg!(target_os = "linux") {
            assert!(!is_permitted_overlap("_ZN6icu_7713UnicodeStringD1Ev"));
        }
    }

    #[test]
    fn mixed_icu_overlap_is_rejected_before_namespacing() {
        let symbols = BTreeSet::from(["_ZN6icu_7613UnicodeStringD1Ev".to_owned()]);
        assert!(reject_mixed_icu_versions(&symbols).is_err());
    }
}
