//! Explicit configuration-scoped C authority. Draft contract: C_VALIDATION_CONTRACT.md.
//! No implicit compiler/configuration selection; no claims about inactive variants.
use super::*;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

const SCHEMA: &str = "greppy.c-validation-result.v1";
const MAX_BYTES: u64 = 64 * 1024 * 1024;
const MAX_FILES: usize = 2048;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Context {
    schema_version: String,
    configuration: String,
    compiler: Compiler,
    translation_units: Vec<Unit>,
    input_files: Vec<String>,
    #[serde(default = "default_timeout")]
    timeout_seconds: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Compiler {
    family: String,
    executable: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Unit {
    file: String,
    directory: String,
    arguments: Vec<String>,
}
fn default_timeout() -> u64 {
    60
}

pub(super) struct Proposed<'a> {
    pub path: &'a str,
    pub before: Option<&'a [u8]>,
    pub after: Option<&'a [u8]>,
}
#[derive(Clone, PartialEq, Eq)]
enum Snapshot {
    Absent,
    Present(String),
}
pub(super) struct Validated {
    _stage: tempfile::TempDir,
    root: PathBuf,
    bindings: BTreeMap<PathBuf, Snapshot>,
    proposals: BTreeMap<String, Snapshot>,
    executable: PathBuf,
    resolved_executable: PathBuf,
    report: serde_json::Value,
}
pub(super) fn is_c(path: &str) -> bool {
    path.ends_with(".c") || path.ends_with(".h")
}
fn base() -> serde_json::Value {
    serde_json::json!({"schema_version":SCHEMA,"backend":"c-compiler","coverage":"selected-configuration","inactive_variants_checked":false})
}
fn fail(
    report: &serde_json::Value,
    status: &str,
    reason: &str,
    message: impl Into<String>,
    exit: i32,
) -> EditRefusal {
    let message = message.into();
    let mut report = report.clone();
    report["status"] = status.into();
    report["reason"] = reason.into();
    report["published"] = false.into();
    EditRefusal::new(
        if status == "stale" {
            "validation_stale"
        } else if status == "rejected" {
            "validation_rejected"
        } else {
            "validation_unavailable"
        },
        format!("C validation {status}: {message}; nothing written"),
        exit,
    )
    .with("c_validation", report)
}
fn relative(text: &str) -> bool {
    !text.is_empty()
        && (text == "." || !text.split('/').any(|part| part.is_empty() || part == "."))
        && !text.contains(['\0', '\n', '\r'])
        && Path::new(text)
            .components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
}
fn project_path(root: &Path, text: &str, report: &serde_json::Value) -> EditResult<PathBuf> {
    if !relative(text) {
        return Err(fail(
            report,
            "unavailable",
            "unsupported_path",
            format!("expected a project-relative path: {text}"),
            14,
        ));
    }
    let mut path = root.to_path_buf();
    for part in Path::new(text).components() {
        path.push(part);
        match std::fs::symlink_metadata(&path) {
            Ok(meta) if meta.file_type().is_symlink() => {
                return Err(fail(
                    report,
                    "unavailable",
                    "symlink_input",
                    format!("input is a symlink: {text}"),
                    14,
                ))
            }
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                return Err(fail(
                    report,
                    "unavailable",
                    "input_unreadable",
                    e.to_string(),
                    14,
                ))
            }
            _ => {}
        }
    }
    Ok(path)
}
fn read_bounded(path: &Path, limit: u64) -> std::io::Result<Vec<u8>> {
    if !std::fs::metadata(path)?.is_file() {
        return Err(std::io::Error::other("input is not a regular file"));
    }
    let file = std::fs::File::open(path)?;
    if !file.metadata()?.is_file() {
        return Err(std::io::Error::other("input is not a regular file"));
    }
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(std::io::Error::other("input exceeds configured byte limit"));
    }
    Ok(bytes)
}
fn digest_file(path: &Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    let meta = std::fs::metadata(path)?;
    if !meta.is_file() {
        return Err(std::io::Error::other(
            "compiler input is not a regular file",
        ));
    }
    let mut file = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn snapshot(path: &Path) -> std::io::Result<Snapshot> {
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_symlink() || !meta.is_file() => Err(std::io::Error::other(
            "input is not a regular nonsymlink file",
        )),
        Ok(_) => digest_file(path).map(Snapshot::Present),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Snapshot::Absent),
        Err(e) => Err(e),
    }
}
fn state(bytes: Option<&[u8]>) -> Snapshot {
    bytes.map_or(Snapshot::Absent, |b| Snapshot::Present(edit_sha256_hex(b)))
}
fn validate_args(unit: &Unit, report: &serde_json::Value) -> EditResult<Vec<String>> {
    if unit.arguments.len() > 128
        || unit
            .arguments
            .iter()
            .filter(|s| s.as_str() == "{source}")
            .count()
            != 1
        || unit
            .arguments
            .iter()
            .filter(|s| s.as_str() == "-fsyntax-only")
            .count()
            != 1
    {
        return Err(fail(
            report,
            "unavailable",
            "unsupported_arguments",
            "exactly one {source} and -fsyntax-only are required",
            14,
        ));
    }
    let mut i = 0;
    while i < unit.arguments.len() {
        let arg = &unit.arguments[i];
        if arg.starts_with("-std=")
            && !matches!(
                arg.as_str(),
                "-std=c89"
                    | "-std=c90"
                    | "-std=iso9899:1990"
                    | "-std=iso9899:199409"
                    | "-std=c99"
                    | "-std=iso9899:1999"
                    | "-std=c11"
                    | "-std=iso9899:2011"
                    | "-std=c17"
                    | "-std=c18"
                    | "-std=iso9899:2017"
                    | "-std=c2x"
                    | "-std=c23"
                    | "-std=gnu89"
                    | "-std=gnu90"
                    | "-std=gnu99"
                    | "-std=gnu11"
                    | "-std=gnu17"
                    | "-std=gnu18"
                    | "-std=gnu2x"
                    | "-std=gnu23"
            )
        {
            return Err(fail(
                report,
                "unavailable",
                "unsupported_arguments",
                "only declared C language standards are supported",
                14,
            ));
        }
        if arg.contains(['\0', '\n', '\r']) {
            return Err(fail(
                report,
                "unavailable",
                "unsupported_arguments",
                "control characters in compiler arguments",
                14,
            ));
        }
        if matches!(arg.as_str(), "-I" | "-iquote" | "-isystem") {
            i += 1;
            if !unit.arguments.get(i).is_some_and(|s| relative(s)) {
                return Err(fail(
                    report,
                    "unavailable",
                    "unsupported_include_path",
                    "include paths must be represented relative paths",
                    14,
                ));
            }
        } else if arg.starts_with("-I") && arg.len() > 2 {
            if !relative(&arg[2..]) {
                return Err(fail(
                    report,
                    "unavailable",
                    "unsupported_include_path",
                    arg,
                    14,
                ));
            }
        } else if !(matches!(
            arg.as_str(),
            "{source}"
                | "-fsyntax-only"
                | "-Wall"
                | "-Wextra"
                | "-Werror"
                | "-pedantic"
                | "-pedantic-errors"
                | "-ffreestanding"
                | "-fno-builtin"
                | "-mthumb"
                | "-m32"
                | "-m64"
                | "-O0"
                | "-O1"
                | "-O2"
                | "-O3"
                | "-Os"
                | "-Og"
                | "-Oz"
        ) || [
            "-D",
            "-U",
            "-std=",
            "--target=",
            "-mcpu=",
            "-march=",
            "-mfpu=",
            "-mfloat-abi=",
            "-mabi=",
        ]
        .iter()
        .any(|prefix| arg.starts_with(prefix) && arg.len() > prefix.len()))
        {
            return Err(fail(report,"unavailable","unsupported_arguments",format!("unsupported frontend argument {arg}; response files, other sources and driver/output modes are not accepted"),14));
        }
        i += 1;
    }
    Ok(unit.arguments.clone())
}
fn dependencies(
    path: &Path,
    cwd: &Path,
    report: &serde_json::Value,
) -> EditResult<BTreeSet<PathBuf>> {
    if std::fs::metadata(path)
        .map_err(|e| {
            fail(
                report,
                "unavailable",
                "dependency_manifest_missing",
                e.to_string(),
                14,
            )
        })?
        .len()
        > 1024 * 1024
    {
        return Err(fail(
            report,
            "unavailable",
            "dependency_manifest_limit",
            "dependency manifest exceeds 1MiB",
            14,
        ));
    }
    let text = std::fs::read_to_string(path).map_err(|e| {
        fail(
            report,
            "unavailable",
            "dependency_manifest_missing",
            e.to_string(),
            14,
        )
    })?;
    if text.len() > 1024 * 1024 {
        return Err(fail(
            report,
            "unavailable",
            "dependency_manifest_limit",
            "dependency manifest exceeds 1MiB",
            14,
        ));
    }
    let text = text.replace("\\\r\n", "").replace("\\\n", "");
    let (_, rhs) = text.split_once(':').ok_or_else(|| {
        fail(
            report,
            "unavailable",
            "dependency_manifest_invalid",
            "missing dependency target",
            14,
        )
    })?;
    let mut words = Vec::new();
    let mut word = String::new();
    let mut chars = rhs.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' => word.push(chars.next().ok_or_else(|| {
                fail(
                    report,
                    "unavailable",
                    "dependency_manifest_invalid",
                    "trailing escape",
                    14,
                )
            })?),
            '$' if chars.peek() == Some(&'$') => {
                chars.next();
                word.push('$');
            }
            '$' | '#' => {
                return Err(fail(
                    report,
                    "unavailable",
                    "dependency_manifest_invalid",
                    "unrepresented make expansion/comment",
                    14,
                ))
            }
            c if c.is_whitespace() => {
                if !word.is_empty() {
                    words.push(std::mem::take(&mut word));
                }
            }
            _ => word.push(c),
        }
    }
    if !word.is_empty() {
        words.push(word);
    }
    if words.is_empty() || words.len() > MAX_FILES {
        return Err(fail(
            report,
            "unavailable",
            "dependency_manifest_limit",
            "empty or excessive dependency manifest",
            14,
        ));
    }
    words
        .into_iter()
        .map(|name| {
            let path = PathBuf::from(name);
            let path = if path.is_absolute() {
                path
            } else {
                cwd.join(path)
            };
            path.canonicalize().map_err(|e| {
                fail(
                    report,
                    "unavailable",
                    "dependency_missing",
                    e.to_string(),
                    14,
                )
            })
        })
        .collect()
}
fn execution(
    program: &Path,
    args: &[String],
    cwd: &Path,
    deadline: Instant,
    report: &serde_json::Value,
) -> EditResult<VerifierExecution> {
    let remaining = deadline
        .checked_duration_since(Instant::now())
        .ok_or_else(|| {
            fail(
                report,
                "timed_out",
                "deadline",
                "transaction validation deadline expired",
                14,
            )
        })?;
    let checked = edit_run_verifier(
        &EditVerifier {
            scope: "c_validation",
            label: "declared C compiler".into(),
            program: program.to_path_buf(),
            args: args.iter().map(Into::into).collect(),
            cwd: cwd.to_path_buf(),
        },
        remaining,
    );
    if checked.status == EditVerificationStatus::TimedOut {
        return Err(fail(
            report,
            "timed_out",
            "compiler_timeout",
            checked.diagnostics.join("\n"),
            14,
        ));
    }
    checked.execution.ok_or_else(|| {
        fail(
            report,
            "unavailable",
            "compiler_unavailable",
            checked.diagnostics.join("\n"),
            14,
        )
    })
}
fn compiler_failure(report: &serde_json::Value, exec: &VerifierExecution) -> EditRefusal {
    let text = format!("{}\n{}", exec.stderr, exec.stdout);
    let lower = text.to_ascii_lowercase();
    let (status, reason) =
        if lower.contains("no such file or directory") || lower.contains("file not found") {
            ("unavailable", "missing_input")
        } else if lower.contains("unknown argument")
            || lower.contains("unrecognized command-line")
            || lower.contains("unsupported option")
            || lower.contains("unknown target")
            || lower.contains("unknown cpu")
        {
            ("unavailable", "unsupported_configuration")
        } else if matches!(exec.exit_code, Some(126 | 127))
            || lower.contains("internal compiler error")
            || lower.contains("unable to execute")
            || lower.contains("cannot execute")
            || lower.contains("error trying to exec")
        {
            ("unavailable", "compiler_execution_failed")
        } else if exec.exit_code.is_none() {
            ("unavailable", "compiler_terminated")
        } else {
            ("rejected", "compiler_rejected")
        };
    let mut evidence = report.clone();
    evidence["process"] = serde_json::json!({"exit_code":exec.exit_code,"timed_out":false,"stderr":exec.stderr,"stdout":exec.stdout});
    fail(&evidence, status, reason, text, 14)
}
pub(super) fn prepare(
    root: &Path,
    context: Option<&str>,
    proposed: &[Proposed<'_>],
    allow_syntax_errors: bool,
) -> EditResult<Option<Validated>> {
    match context {
        None => Ok(None),
        Some(_) if allow_syntax_errors => Err(fail(
            &base(),
            "unavailable",
            "conflicting_options",
            "--c-context cannot be combined with --allow-syntax-errors",
            20,
        )),
        Some(file) => validate(root, file, proposed).map(Some),
    }
}
fn validate(root: &Path, file: &str, proposed: &[Proposed<'_>]) -> EditResult<Validated> {
    let mut report = base();
    if !cfg!(unix) {
        return Err(fail(
            &report,
            "unavailable",
            "unsupported_platform",
            "native C contexts currently require Unix GCC/Clang",
            14,
        ));
    }
    let root = root.canonicalize().map_err(|e| {
        fail(
            &report,
            "unavailable",
            "workspace_unavailable",
            e.to_string(),
            14,
        )
    })?;
    let context_path = project_path(&root, file, &report)?;
    if std::fs::metadata(&context_path)
        .map_err(|e| fail(&report, "unavailable", "context_missing", e.to_string(), 14))?
        .len()
        > 1024 * 1024
    {
        return Err(fail(
            &report,
            "unavailable",
            "context_limit",
            "context exceeds 1MiB",
            14,
        ));
    }
    let context_bytes = read_bounded(&context_path, 1024 * 1024)
        .map_err(|e| fail(&report, "unavailable", "context_missing", e.to_string(), 14))?;
    if context_bytes.len() > 1024 * 1024 {
        return Err(fail(
            &report,
            "unavailable",
            "context_limit",
            "context exceeds 1MiB",
            14,
        ));
    }
    report["context_sha256"] = edit_sha256_hex(&context_bytes).into();
    let context: Context = serde_json::from_slice(&context_bytes)
        .map_err(|e| fail(&report, "unavailable", "invalid_context", e.to_string(), 20))?;
    if context.schema_version != "greppy.c-validation-context.v1"
        || context.configuration.trim().is_empty()
        || !matches!(context.compiler.family.as_str(), "gcc" | "clang")
        || context.translation_units.is_empty()
        || context.translation_units.len() > 32
        || context.input_files.len() > MAX_FILES
        || context.timeout_seconds == 0
        || context.timeout_seconds > 60
    {
        return Err(fail(
            &report,
            "unavailable",
            "unsupported_context",
            "unsupported schema, compiler family, configuration, limits or timeout",
            14,
        ));
    }
    report["configuration"] = context.configuration.clone().into();
    let executable = PathBuf::from(&context.compiler.executable);
    if !executable.is_absolute() {
        return Err(fail(
            &report,
            "unavailable",
            "compiler_path",
            "compiler executable must be explicit and absolute",
            14,
        ));
    }
    let resolved = executable.canonicalize().map_err(|e| {
        fail(
            &report,
            "unavailable",
            "compiler_missing",
            e.to_string(),
            14,
        )
    })?;
    let compiler_hash = digest_file(&resolved).map_err(|e| {
        fail(
            &report,
            "unavailable",
            "compiler_unreadable",
            e.to_string(),
            14,
        )
    })?;
    report["compiler"] = serde_json::json!({"family":context.compiler.family,"path":executable,"resolved_path":resolved,"sha256":compiler_hash});
    let deadline =
        Instant::now() + Duration::from_secs(context.timeout_seconds).min(edit_verify_timeout());
    let version = execution(&resolved, &["--version".into()], &root, deadline, &report)?;
    if version.exit_code != Some(0) {
        return Err(compiler_failure(&report, &version));
    }
    let lower = version.stdout.to_ascii_lowercase();
    if !(if context.compiler.family == "clang" {
        lower.contains("clang")
    } else {
        lower.contains("gcc") && !lower.contains("clang")
    }) {
        return Err(fail(
            &report,
            "unavailable",
            "compiler_family_mismatch",
            "compiler version does not match declared family",
            14,
        ));
    }
    report["compiler"]["version"] = version.stdout.lines().next().unwrap_or_default().into();
    let stage = tempfile::Builder::new()
        .prefix("greppy-c-validation-")
        .tempdir()
        .map_err(|e| {
            fail(
                &report,
                "unavailable",
                "stage_unavailable",
                e.to_string(),
                14,
            )
        })?;
    let staged = stage.path().join("workspace");
    std::fs::create_dir(&staged).map_err(|e| {
        fail(
            &report,
            "unavailable",
            "stage_unavailable",
            e.to_string(),
            14,
        )
    })?;
    let mut files: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let mut bindings = BTreeMap::from([
        (
            context_path.clone(),
            Snapshot::Present(edit_sha256_hex(&context_bytes)),
        ),
        (resolved.clone(), Snapshot::Present(compiler_hash)),
    ]);
    let mut input_names = BTreeSet::new();
    for name in &context.input_files {
        if !input_names.insert(name.clone()) {
            return Err(fail(&report, "unavailable", "ambiguous_input", name, 20));
        }
        let path = project_path(&root, name, &report)?;
        if proposed.iter().any(|p| p.path == name) {
            continue;
        }
        let metadata = std::fs::metadata(&path)
            .map_err(|e| fail(&report, "unavailable", "input_missing", e.to_string(), 14))?;
        if !metadata.is_file() || metadata.len() > MAX_BYTES {
            return Err(fail(
                &report,
                "unavailable",
                "input_limit",
                format!("input not representable: {name}"),
                14,
            ));
        }
        let bytes = read_bounded(&path, MAX_BYTES)
            .map_err(|e| fail(&report, "unavailable", "input_missing", e.to_string(), 14))?;
        bindings.insert(path, Snapshot::Present(edit_sha256_hex(&bytes)));
        files.insert(name.clone(), bytes);
        if files.values().map(|b| b.len() as u64).sum::<u64>() > MAX_BYTES {
            return Err(fail(
                &report,
                "unavailable",
                "stage_limit",
                "declared snapshot exceeds64MiB",
                14,
            ));
        }
    }
    let mut proposal_states = BTreeMap::new();
    let mut changed_c = BTreeSet::new();
    let mut parser = Vec::new();
    for item in proposed {
        let path = project_path(&root, item.path, &report)?;
        if path == context_path {
            return Err(fail(
                &report,
                "unavailable",
                "context_edited",
                "select a stable context outside the proposed change",
                14,
            ));
        }
        if proposal_states
            .insert(item.path.to_string(), state(item.after))
            .is_some()
        {
            return Err(fail(
                &report,
                "unavailable",
                "duplicate_proposal",
                item.path,
                20,
            ));
        }
        bindings.insert(path, state(item.before));
        if is_c(item.path) {
            if item.after.is_none() {
                return Err(fail(
                    &report,
                    "unavailable",
                    "deleted_c_input",
                    "native v1 does not certify deletion of a C input",
                    14,
                ));
            }
            changed_c.insert(item.path.to_string());
            let bytes = item.after.unwrap_or_default();
            let tree =
                greppy_parser::parse_for_syntax_validation(greppy_parser::Language::C, bytes)
                    .map_err(|e| {
                        fail(
                            &report,
                            "unavailable",
                            "parser_observation_unavailable",
                            e.to_string(),
                            14,
                        )
                    })?;
            {
                let mut nodes = vec![tree.root_node()];
                let (mut errors, mut missing) = (0, 0);
                while let Some(node) = nodes.pop() {
                    errors += usize::from(node.is_error());
                    missing += usize::from(node.is_missing());
                    let mut cursor = node.walk();
                    nodes.extend(node.children(&mut cursor));
                }
                parser.push(serde_json::json!({"path":item.path,"role":"observation","errors":errors,"missing":missing}));
            }
        }
        if let Some(bytes) = item.after {
            if bytes.len() as u64 > MAX_BYTES
                || files
                    .iter()
                    .filter(|(p, _)| p.as_str() != item.path)
                    .map(|(_, b)| b.len() as u64)
                    .sum::<u64>()
                    + bytes.len() as u64
                    > MAX_BYTES
            {
                return Err(fail(
                    &report,
                    "unavailable",
                    "stage_limit",
                    "proposed snapshot exceeds64MiB",
                    14,
                ));
            }
        }
        match item.after {
            Some(bytes) => {
                files.insert(item.path.to_string(), bytes.to_vec());
            }
            None => {
                files.remove(item.path);
            }
        }
    }
    if changed_c.is_empty() {
        return Err(fail(
            &report,
            "unavailable",
            "no_c_proposal",
            "context option requires a .c or .h proposal",
            14,
        ));
    }
    if files.len() > MAX_FILES || files.values().map(|b| b.len() as u64).sum::<u64>() > MAX_BYTES {
        return Err(fail(
            &report,
            "unavailable",
            "stage_limit",
            "declared snapshot exceeds 2048 files/64MiB",
            14,
        ));
    }
    for (name, bytes) in &files {
        if !relative(name) {
            return Err(fail(&report, "unavailable", "unsupported_path", name, 14));
        }
        let target = staged.join(name);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                fail(
                    &report,
                    "unavailable",
                    "stage_unavailable",
                    e.to_string(),
                    14,
                )
            })?;
        }
        std::fs::write(target, bytes).map_err(|e| {
            fail(
                &report,
                "unavailable",
                "stage_unavailable",
                e.to_string(),
                14,
            )
        })?;
    }
    report["raw_parser_observations"] = parser.into();
    report["input_hashes"] = serde_json::to_value(
        files
            .iter()
            .map(|(p, b)| (p, edit_sha256_hex(b)))
            .collect::<BTreeMap<_, _>>(),
    )
    .unwrap_or_default();
    let mut seen = BTreeSet::new();
    let mut covered = BTreeSet::new();
    let mut units = Vec::new();
    for (index, unit) in context.translation_units.iter().enumerate() {
        if !unit.file.ends_with(".c")
            || !relative(&unit.file)
            || !relative(&unit.directory)
            || !files.contains_key(&unit.file)
            || !seen.insert(unit.file.clone())
        {
            return Err(fail(&report,"unavailable","unknown_translation_unit","each unique .c unit must be a proposal or declared input, with a represented directory",14));
        }
        let cwd = staged.join(&unit.directory);
        std::fs::create_dir_all(&cwd).map_err(|e| {
            fail(
                &report,
                "unavailable",
                "stage_unavailable",
                e.to_string(),
                14,
            )
        })?;
        let args = validate_args(unit, &report)?
            .into_iter()
            .map(|a| {
                if a == "{source}" {
                    staged.join(&unit.file).to_string_lossy().into_owned()
                } else {
                    a
                }
            })
            .collect::<Vec<_>>();
        let dep = stage.path().join(format!("before-{index}.d"));
        let mut scan = args
            .iter()
            .filter(|a| a.as_str() != "-fsyntax-only")
            .cloned()
            .collect::<Vec<_>>();
        scan.extend([
            "-M".into(),
            "-MF".into(),
            dep.to_string_lossy().into_owned(),
            "-MT".into(),
            "greppy".into(),
        ]);
        let result = execution(&resolved, &scan, &cwd, deadline, &report)?;
        if result.exit_code != Some(0) {
            return Err(compiler_failure(&report, &result));
        }
        let deps = dependencies(&dep, &cwd, &report)?;
        let mut dependency_hashes = BTreeMap::new();
        for path in &deps {
            if let Ok(relative) = path.strip_prefix(&staged) {
                let name = relative.to_string_lossy().replace('\\', "/");
                if !files.contains_key(&name) {
                    return Err(fail(
                        &report,
                        "unavailable",
                        "unrepresented_input",
                        name,
                        14,
                    ));
                }
                covered.insert(name);
            } else if path.starts_with(&root) {
                return Err(fail(
                    &report,
                    "unavailable",
                    "live_input_escape",
                    "compiler input bypasses the staged transaction",
                    14,
                ));
            }
            let hash = digest_file(path).map_err(|e| {
                fail(
                    &report,
                    "unavailable",
                    "dependency_unreadable",
                    e.to_string(),
                    14,
                )
            })?;
            dependency_hashes.insert(path.clone(), hash);
        }
        let post = stage.path().join(format!("after-{index}.d"));
        let mut check = vec!["-x".into(), "c".into()];
        check.extend(args.clone());
        check.extend([
            "-MD".into(),
            "-MF".into(),
            post.to_string_lossy().into_owned(),
            "-MT".into(),
            "greppy".into(),
        ]);
        let checked = execution(&resolved, &check, &cwd, deadline, &report)?;
        if checked.exit_code != Some(0) {
            return Err(compiler_failure(&report, &checked));
        }
        if dependencies(&post, &cwd, &report)? != deps {
            return Err(fail(
                &report,
                "stale",
                "dependencies_changed",
                "compiler dependency set changed",
                12,
            ));
        }
        for (path, hash) in &dependency_hashes {
            if digest_file(path).ok().as_ref() != Some(hash) {
                return Err(fail(
                    &report,
                    "stale",
                    "input_changed",
                    "compiler input changed while checking",
                    12,
                ));
            }
            if !path.starts_with(&staged) {
                bindings.insert(path.clone(), Snapshot::Present(hash.clone()));
            }
        }
        units.push(serde_json::json!({"file":unit.file,"source_sha256":edit_sha256_hex(&files[&unit.file]),"directory":unit.directory,"arguments":check,"process":{"exit_code":checked.exit_code,"timed_out":false},"dependency_hashes":dependency_hashes}));
    }
    if !changed_c.is_subset(&covered) {
        return Err(fail(
            &report,
            "unavailable",
            "uncovered_proposal",
            "a changed C/header input was not consumed by any declared translation unit",
            14,
        ));
    }
    report["translation_units"] = units.into();
    report["status"] = "passed".into();
    let validated = Validated {
        _stage: stage,
        root,
        bindings,
        proposals: proposal_states,
        executable,
        resolved_executable: resolved,
        report,
    };
    validated.recheck(proposed)?;
    Ok(validated)
}
impl Validated {
    pub(super) fn recheck(&self, proposed: &[Proposed<'_>]) -> EditResult<()> {
        self.recheck_progress(proposed, 0)
    }
    pub(super) fn recheck_progress(
        &self,
        proposed: &[Proposed<'_>],
        published_count: usize,
    ) -> EditResult<()> {
        let published = proposed
            .iter()
            .take(published_count)
            .map(|p| (self.root.join(p.path), state(p.after)))
            .collect::<BTreeMap<_, _>>();
        if self.executable.canonicalize().ok().as_ref() != Some(&self.resolved_executable) {
            return Err(fail(
                &self.report,
                "stale",
                "compiler_changed",
                "compiler selection changed",
                12,
            ));
        }
        for (path, before) in &self.bindings {
            let expected = published.get(path).unwrap_or(before);
            if snapshot(path).ok().as_ref() != Some(expected) {
                return Err(fail(
                    &self.report,
                    "stale",
                    "input_changed",
                    format!("validated input changed: {}", path.display()),
                    12,
                ));
            }
        }
        let current = proposed
            .iter()
            .map(|p| (p.path.to_string(), state(p.after)))
            .collect::<BTreeMap<_, _>>();
        if current != self.proposals {
            return Err(fail(
                &self.report,
                "stale",
                "proposal_changed",
                "proposed bytes changed after validation",
                12,
            ));
        }
        Ok(())
    }
    pub(super) fn receipt(&self, published: bool) -> serde_json::Value {
        let mut report = self.report.clone();
        report["published"] = published.into();
        report
    }
}
