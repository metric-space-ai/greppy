use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const IRREGEXP_MEMBER: &str = "Unified_cpp_js_src_irregexp0.o";

/// V8/SpiderMonkey irregexp overlap: ld64 coalesces these and V8 Isolate::Delete
/// runs the SpiderMonkey destructor (KERN_INVALID_ADDRESS).
///
/// `nmedit -R` only turns SM copies into "non-external (was a private external)".
/// ld64 still coalesces those with V8's private-extern destructors. Proven crash:
/// `Isolate::Delete` → `Deinitialize` → SM `Isolate::~Isolate` (`mozilla::SegmentedVector`).
/// Rename the SM symbol *strings* (same length) so the final bind cannot pick them.
/// Intra-object relocations follow the symbol index, not the string, so SM still
/// calls its own destructor. Keep js::irregexp::* exported.
const HIDE_SYMBOLS: &[&str] = &[
    "__ZN2v88internal7IsolateD1Ev",
    "__ZN2v88internal7IsolateD2Ev",
    "__ZN2v88internal6PrintFEPKcz",
    "__ZN2v88internal6PrintFEP7__sFILEPKcz",
];

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

pub fn localize_mozjs_js_static() {
    if cfg!(windows) {
        panic!(
            "web-runtime one-binary localization is an explicit unsatisfied gate on Windows"
        );
    }
    let archives = find_js_static_archives();
    if archives.is_empty() {
        panic!("mozjs_sys libjs_static.a not found; cannot localize V8/SpiderMonkey overlap");
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
    let Some(v8) = find_rusty_v8() else {
        panic!("librusty_v8.a not found; cannot run defined-symbol intersection gate");
    };
    println!("cargo:rerun-if-changed={}", v8.display());
    for archive in &archives {
        if let Err(error) = assert_symbol_intersection(archive, &v8) {
            panic!("symbol intersection {}: {error}", archive.display());
        }
    }
}

fn find_mozjs_sys_rlibs() -> Vec<PathBuf> {
    let mut found = Vec::new();
    for root in search_roots() {
        let deps = root.join("deps");
        let Ok(entries) = fs::read_dir(&deps) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("libmozjs_sys-") && name.ends_with(".rlib") {
                let path = entry.path();
                if path.is_file() && !found.contains(&path) {
                    found.push(path);
                }
            }
        }
    }
    found
}

fn find_js_static_archives() -> Vec<PathBuf> {
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
                .join("libjs_static.a");
            if archive.is_file() && !found.contains(&archive) {
                found.push(archive);
            }
        }
    }
    found
}

fn find_rusty_v8() -> Option<PathBuf> {
    for root in search_roots() {
        let candidate = root.join("gn_out").join("obj").join("librusty_v8.a");
        if candidate.is_file() {
            return Some(candidate);
        }
        if let Ok(entries) = fs::read_dir(root.join("build")) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                if !name.to_string_lossy().starts_with("rusty_v8-") {
                    continue;
                }
                let nested = entry.path().join("out").join("obj").join("librusty_v8.a");
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
        roots.push(dir.join(env::var("TARGET").unwrap_or_default()).join(&profile));
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
    let status = Command::new("ar")
        .args(["x", archive.to_str().unwrap(), IRREGEXP_MEMBER])
        .current_dir(&work)
        .status()
        .map_err(|e| format!("ar x: {e}"))?;
    if !status.success() {
        return Err(format!("ar x failed: {status}"));
    }
    let member = work.join(IRREGEXP_MEMBER);
    if !member.is_file() {
        return Err(format!("missing {IRREGEXP_MEMBER} in {}", archive.display()));
    }
    rename_v8_overlap_symbols(&member)?;
    let archive_str = archive.to_str().unwrap();
    let _ = Command::new("ar")
        .args(["d", archive_str, IRREGEXP_MEMBER])
        .status();
    let status = Command::new("ar")
        .args(["r", archive_str, IRREGEXP_MEMBER])
        .current_dir(&work)
        .status()
        .map_err(|e| format!("ar r: {e}"))?;
    if !status.success() {
        return Err(format!("ar r failed: {status}"));
    }
    if archive.extension().and_then(|e| e.to_str()) != Some("rlib") {
        let _ = Command::new("ranlib").arg(archive).status();
    }
    Ok(())
}

fn verify_hidden(archive: &Path) -> Result<(), String> {
    let defined = defined_symbols(archive)?;
    let still: Vec<_> = HIDE_SYMBOLS
        .iter()
        .copied()
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
    let mut icu = 0usize;
    let mut dangerous = Vec::new();
    for name in sm.intersection(&v8) {
        if is_icu(name) {
            if name.contains("icu_76")
                || name.contains("icudt76")
                || name.contains("_76")
            {
                return Err(format!(
                    "ICU overlap {name} is not ICU 77; refusing mixed ICU versions"
                ));
            }
            icu += 1;
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
    println!("cargo:warning=engine symbol intersection: 0 non-ICU overlaps, {icu} ICU 77 overlaps (coalesced same-version ICU is permitted)");
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
    let output = Command::new("nm")
        .args(&args)
        .output()
        .map_err(|e| format!("nm {}: {e}", archive.display()))?;
    if !output.status.success() {
        return Err(format!(
            "nm {} failed: {}\n{}",
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
    for (from, to) in RENAME_SYMBOLS {
        if from.len() != to.len() {
            return Err(format!("rename length mismatch {from} -> {to}"));
        }
    }
    if cfg!(target_os = "linux") {
        let mut cmd = Command::new("objcopy");
        for (from, to) in RENAME_SYMBOLS {
            cmd.arg(format!("--redefine-sym={from}={to}"));
        }
        let output = cmd
            .arg(object)
            .output()
            .map_err(|e| format!("objcopy redefine: {e}"))?;
        if !output.status.success() {
            return Err(format!(
                "objcopy redefine failed: {}\n{}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        return Ok(());
    }
    let mut bytes = fs::read(object).map_err(|e| format!("read {}: {e}", object.display()))?;
    for (from, to) in RENAME_SYMBOLS {
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
    haystack.windows(needle.len()).position(|window| window == needle)
}
