//! Native Linux/GCC link-time isolation. Dependency archives remain read-only.
#![allow(dead_code)]
include!("localize_js_static.rs");
use std::ffi::{OsStr, OsString};
use std::io::Read;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::os::unix::fs::MetadataExt;

fn digest(path: &Path) -> Result<String, String> {
    let out = Command::new("sha256sum").arg(path).output().map_err(|e| e.to_string())?;
    if !out.status.success() || out.stdout.len() < 64
        || !out.stdout[..64].iter().all(u8::is_ascii_hexdigit) {
        return Err(format!("cannot hash {}", path.display()));
    }
    Ok(String::from_utf8(out.stdout[..64].to_vec()).unwrap())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// GNU response-file syntax, not shell syntax: no expansion, substitutions or comments.
fn response_words(bytes: &[u8]) -> Result<Vec<OsString>, String> {
    let mut result = Vec::new();
    let mut value = Vec::new();
    let (mut quote, mut started, mut i) = (None, false, 0);
    while i < bytes.len() {
        let b = bytes[i];
        if b == 0 { return Err("NUL in response file".into()); }
        if b == b'\\' {
            i += 1;
            if i == bytes.len() { return Err("dangling response-file escape".into()); }
            if bytes[i] == 0 { return Err("NUL in response-file escape".into()); }
            value.push(bytes[i]); started = true;
        } else if quote == Some(b) {
            quote = None;
        } else if quote.is_none() && (b == b'\'' || b == b'"') {
            quote = Some(b); started = true;
        } else if quote.is_none() && b.is_ascii_whitespace() {
            if started { result.push(OsString::from_vec(std::mem::take(&mut value))); started = false; }
        } else {
            value.push(b); started = true;
        }
        i += 1;
    }
    if quote.is_some() { return Err("unterminated response-file quote".into()); }
    if started { result.push(OsString::from_vec(value)); }
    Ok(result)
}

fn expand(args: Vec<OsString>, depth: usize, remaining: &mut u64) -> Result<Vec<OsString>, String> {
    if depth > 8 { return Err("response-file nesting exceeds eight".into()); }
    let mut result = Vec::new();
    for arg in args {
        let bytes = arg.as_bytes();
        if bytes.first() == Some(&b'@') {
            let file = PathBuf::from(OsString::from_vec(bytes[1..].to_vec()));
            let metadata = fs::symlink_metadata(&file).map_err(|e| format!("response file: {e}"))?;
            if !metadata.is_file() || metadata.len() > 4 * 1024 * 1024 || metadata.len() > *remaining {
                return Err("response file must be a regular file of at most 4 MiB".into());
            }
            let input = fs::File::open(&file).map_err(|e| e.to_string())?;
            let mut bytes = Vec::new();
            input.take(metadata.len()+1).read_to_end(&mut bytes).map_err(|e|e.to_string())?;
            if bytes.len() as u64 != metadata.len() {
                return Err("response file changed while reading".into());
            }
            *remaining -= metadata.len();
            result.extend(expand(response_words(&bytes)?, depth + 1, remaining)?);
        } else { result.push(arg); }
        if result.len() > 100_000 { return Err("expanded linker argument bound exceeded".into()); }
    }
    Ok(result)
}

fn response_bytes(args: &[OsString]) -> Vec<u8> {
    let mut out = Vec::new();
    for arg in args {
        out.push(b'"');
        for b in arg.as_bytes() {
            if *b == b'\\' || *b == b'"' { out.push(b'\\'); }
            out.push(*b);
        }
        out.extend_from_slice(b"\"\n");
    }
    out
}

fn selected(args: &[OsString], prefix: &str) -> Vec<PathBuf> {
    args.iter().filter_map(|arg| {
        let p = PathBuf::from(arg);
        let name = p.file_name()?.to_str()?;
        (name.starts_with(prefix) && name.ends_with(".rlib")).then_some(p)
    }).collect::<BTreeSet<_>>().into_iter().collect()
}

fn identity(path: &Path) -> Result<(u64,u64,u64,i64,i64,i64,i64),String> {
    let s = fs::metadata(path).map_err(|e| e.to_string())?;
    Ok((s.dev(),s.ino(),s.len(),s.mtime(),s.mtime_nsec(),s.ctime(),s.ctime_nsec()))
}

fn run_late() -> Result<i32,String> {
    let mut input = env::args_os().skip(1);
    // GCC supplies the actual subprogram; never guess cc, ld or rust-lld.
    let driver = input.next().ok_or("missing GCC subprogram")?;
    let passed = input.collect::<Vec<_>>();
    let original = expand(passed.clone(),0,&mut (8 * 1024 * 1024))?;
    let sm = selected(&original,"libmozjs_sys-");
    let v8 = selected(&original,"libv8-");
    if sm.is_empty() || v8.is_empty() {
        return delegate(&driver,&passed);
    }
    if sm.len()!=1 || v8.len()!=1 {
        return Err("combined link must select exactly one mozjs_sys and one v8 rlib".into());
    }
    let old_diplomat = selected(&original,"libdiplomat_runtime-").into_iter()
        .map(|p| defined_globals(&p).map(|s|(p,s)))
        .collect::<Result<Vec<_>,_>>()?.into_iter()
        .filter_map(|(p,s)| s.contains("diplomat_buffer_write_create").then_some(p))
        .collect::<Vec<_>>();
    let capi = selected(&original,"libicu_capi-");
    if old_diplomat.len()!=1 || capi.is_empty() {
        return Err("selected old Diplomat ABI and ICU C API are required".into());
    }
    let parent = env::current_exe().map_err(|e|e.to_string())?.parent()
        .ok_or("adapter has no OUT_DIR parent")?.to_owned();
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map_err(|e|e.to_string())?.as_nanos();
    let work = parent.join(format!("namespace-link-{}-{stamp}",std::process::id()));
    fs::create_dir(&work).map_err(|e|e.to_string())?;
    let mut copies = std::collections::BTreeMap::new();
    let mut originals = Vec::new();
    for path in sm.iter().chain(old_diplomat.iter()).chain(capi.iter()).chain(v8.iter()) {
        let before = identity(path)?;
        let hash = digest(path)?;
        originals.push((path.clone(),before,hash.clone()));
        if v8.contains(path) { continue; }
        let copy = work.join(path.file_name().ok_or("archive has no filename")?);
        fs::copy(path,&copy).map_err(|e|e.to_string())?;
        if digest(&copy)?!=hash || identity(path)?!=before {
            return Err("archive changed during private copy".into());
        }
        copies.insert(path.clone(),copy);
    }
    let private_sm = &copies[&sm[0]];
    localize_archive(private_sm)?;
    let v8_globals = defined_globals(&v8[0])?;
    let globals = defined_globals(private_sm)?;
    let overlaps = globals.intersection(&v8_globals).filter(|s|is_icu(s))
        .cloned().collect::<BTreeSet<_>>();
    reject_mixed_icu_versions(&overlaps)?;
    let comdat = local_icu_comdat_signatures(private_sm,"__greppy_sm_")?;
    reject_mixed_icu_versions(&comdat)?;
    redefine_linux_archive_symbols(private_sm,&overlaps.union(&comdat).cloned().collect(),"__greppy_sm_")?;
    verify_symbols_absent(private_sm,&overlaps,"ICU")?;
    verify_local_icu_comdat_absent(private_sm,"__greppy_sm_")?;
    verify_symbols_renamed(private_sm,&overlaps,"__greppy_sm_","ICU")?;
    let allocators = BTreeSet::from(["diplomat_alloc".to_owned(),"diplomat_free".to_owned()]);
    for path in sm.iter().chain(old_diplomat.iter()).chain(capi.iter()) {
        let copy = &copies[path];
        let pending = pending_symbols(&allocators,&archive_symbol_names(copy)?);
        redefine_linux_archive_symbols(copy,&pending,"greppy_sm_")?;
        if !pending_symbols(&allocators,&archive_symbol_names(copy)?).is_empty() {
            return Err("old Diplomat allocator names remain in a private archive".into());
        }
    }
    verify_symbols_absent(&copies[&old_diplomat[0]],&allocators,"diplomat-runtime")?;
    assert_symbol_intersection(private_sm,&v8[0])?;
    let final_args = original.iter().map(|arg| copies.get(&PathBuf::from(arg))
        .map(|p|p.as_os_str().to_owned()).unwrap_or_else(||arg.clone())).collect::<Vec<_>>();
    let response = response_bytes(&final_args);
    if response_words(&response)?!=final_args { return Err("response encoding changed arguments".into()); }
    let response_path = work.join("linker.rsp");
    fs::write(&response_path,response).map_err(|e|e.to_string())?;
    let mut ledger = format!("driver_path_hex={}\n",hex(driver.as_bytes()));
    for (path,before,hash) in &originals {
        ledger.push_str(&format!("input_path_hex={} identity={before:?} sha256={hash}\n",hex(path.as_os_str().as_bytes())));
        if let Some(copy)=copies.get(path) {
            ledger.push_str(&format!("private_path_hex={} sha256={}\n",hex(copy.as_os_str().as_bytes()),digest(copy)?));
        }
    }
    fs::write(work.join("inputs.txt"),&ledger).map_err(|e|e.to_string())?;
    for (path,before,hash) in &originals {
        if identity(path)?!=*before || digest(path)?!=*hash { return Err("dependency changed before link".into()); }
    }
    eprintln!("GREPPY_LATE_NAMESPACE_INPUTS={}",work.join("inputs.txt").display());
    let response_arg = {
        let mut bytes = vec![b'@']; bytes.extend_from_slice(response_path.as_os_str().as_bytes());
        OsString::from_vec(bytes)
    };
    let code = delegate(&driver,&[response_arg])?;
    for (path,before,hash) in &originals {
        if identity(path)?!=*before || digest(path)?!=*hash { return Err("dependency changed during link".into()); }
    }
    ledger.push_str(&format!("link_exit={code}\ndependency_archives_unchanged=true\n"));
    fs::write(work.join("inputs.txt"),ledger).map_err(|e|e.to_string())?;
    Ok(code)
}

fn delegate(driver: &OsStr,args: &[OsString]) -> Result<i32,String> {
    use std::os::unix::process::ExitStatusExt;
    let status = Command::new(driver).args(args).status().map_err(|e|e.to_string())?;
    Ok(status.code().unwrap_or_else(||128+status.signal().unwrap_or(0)))
}

fn main() {
    match run_late() {
        Ok(code)=>std::process::exit(code),
        Err(error)=>{ eprintln!("Greppy link isolation failed: {error}");std::process::exit(1); }
    }
}

#[cfg(test)]
mod late_link_contracts {
    use super::*;

    #[test]
    fn gnu_response_arguments_preserve_bytes_without_shell_expansion() {
        let source = br#""" "a b" 'c d' e\\f \"quote\" "$HOME;$(ignored)" "#;
        let expected = ["", "a b", "c d", r"e\f", "\"quote\"", "$HOME;$(ignored)"]
            .into_iter().map(OsString::from).collect::<Vec<_>>();
        assert_eq!(response_words(source).unwrap(),expected);
        let mut bytes = expected;
        bytes.push(OsString::from_vec(vec![b'x',0xff,b'\n',b'\t']));
        assert_eq!(response_words(&response_bytes(&bytes)).unwrap(),bytes);
        assert!(response_words(b"unterminated\\").is_err());
        assert!(response_words(b"'unterminated").is_err());
    }

    #[test]
    fn archive_selection_never_rewrites_embedded_flag_text() {
        let args = ["/one/libmozjs_sys-ab.rlib","-Wl,/two/libmozjs_sys-zz.rlib",
                    "/one/libmozjs_sys-ab.rlib","/three/not-libmozjs_sys.rlib",
                    "/one/libmozjs_sys-ab.rmeta"].map(OsString::from);
        assert_eq!(selected(&args,"libmozjs_sys-"),vec![PathBuf::from("/one/libmozjs_sys-ab.rlib")]);
    }
}

