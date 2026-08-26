use clap::Parser;
use greppy_workspace_core::{capture_repository, ProviderInstallation, WorkspaceCore, CHUNK_SIZE};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Parser)]
#[command(about = "Fail-closed portable workspace release performance gate")]
struct Args {
    #[arg(long)]
    repository: PathBuf,
    #[arg(long)]
    data_root: PathBuf,
    #[arg(long)]
    probe_path: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    source_commit: String,
    #[arg(long)]
    provider_binary: PathBuf,
    #[arg(long)]
    native_baseline_root: PathBuf,
    #[arg(long)]
    hardware: String,
    #[arg(long, default_value_t = 25)]
    iterations: usize,
    #[arg(long, default_value_t = 5)]
    native_baseline_iterations: usize,
    #[arg(long, default_value_t = 50)]
    parallel: usize,
    #[arg(long, default_value_t = 300_000)]
    expected_files: usize,
    #[arg(long, default_value_t = 500.0)]
    max_visible_p95_ms: f64,
    #[arg(long, default_value_t = 1_048_576)]
    max_untouched_bytes: u64,
    #[arg(long, default_value_t = 1_310_720)]
    max_one_byte_write_bytes: u64,
    #[arg(long)]
    enforce: bool,
    /// JSON: {"name":"rust|python|node","argv":["program","arg",...]}
    #[arg(long = "toolchain-case")]
    toolchain_cases: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ToolchainCase {
    name: String,
    argv: Vec<String>,
    #[serde(default)]
    cwd: PathBuf,
}

#[derive(Debug, Serialize)]
struct ToolchainResult {
    name: String,
    native_ms: f64,
    workspace_ms: f64,
    overhead_percent: f64,
}

fn main() {
    let args = Args::parse();
    if cfg!(debug_assertions) {
        fail("release performance evidence requires --release");
    }
    if !args.repository.is_absolute()
        || !args.data_root.is_absolute()
        || !args.output.is_absolute()
        || !args.provider_binary.is_absolute()
        || !args.native_baseline_root.is_absolute()
    {
        fail("repository, data-root, output and provider-binary must be absolute");
    }
    if args.iterations < 5 || args.parallel == 0 {
        fail("at least five iterations and one parallel workspace are required");
    }
    let provider = ProviderInstallation::require_healthy(&args.data_root)
        .unwrap_or_else(|error| fail(&format!("provider is not healthy: {error}")));
    provider
        .doctor_io("release-performance")
        .unwrap_or_else(|error| fail(&format!("provider I/O doctor failed: {error}")));
    let tracked_files = tracked_file_count(&args.repository);
    if tracked_files != args.expected_files {
        fail(&format!(
            "fixture has {tracked_files} tracked files, expected exactly {}",
            args.expected_files
        ));
    }
    let probe_expected = fs::read(args.repository.join(&args.probe_path))
        .unwrap_or_else(|error| fail(&format!("cannot read native probe path: {error}")));
    let core = WorkspaceCore::open(args.data_root.join("core"))
        .unwrap_or_else(|error| fail(&format!("cannot open WorkspaceCore: {error}")));

    // Prime schemas, SQLite pages and provider path caches. Measurements start
    // only after this workspace has been removed.
    let prime = capture_repository(&args.repository, core.chunks()).unwrap();
    let prime_handle = core.create_workspace("perf-prime", prime).unwrap();
    let prime_root = provider.workspace_path(prime_handle.id()).unwrap();
    assert_eq!(
        fs::read(prime_root.join(&args.probe_path)).unwrap(),
        probe_expected
    );
    core.remove_workspace(prime_handle).unwrap();

    let physical_before = allocated_tree_bytes(&args.data_root);
    let untouched_baseline = capture_repository(&args.repository, core.chunks()).unwrap();
    let untouched = core
        .create_workspace("perf-untouched", untouched_baseline)
        .unwrap();
    let untouched_root = provider.workspace_path(untouched.id()).unwrap();
    assert_eq!(
        fs::read(untouched_root.join(&args.probe_path)).unwrap(),
        probe_expected
    );
    let physical_after = allocated_tree_bytes(&args.data_root);
    let untouched_physical_delta = physical_after.saturating_sub(physical_before);
    core.remove_workspace(untouched).unwrap();

    let mut snapshot_ms = Vec::with_capacity(args.iterations);
    let mut visible_ms = Vec::with_capacity(args.iterations);
    let mut end_to_end_ms = Vec::with_capacity(args.iterations);
    for iteration in 0..args.iterations {
        let end_to_end_started = Instant::now();
        let snapshot_started = Instant::now();
        let baseline = capture_repository(&args.repository, core.chunks()).unwrap();
        snapshot_ms.push(snapshot_started.elapsed().as_secs_f64() * 1_000.0);
        let visible_started = Instant::now();
        let handle = core
            .create_workspace(&format!("perf-serial-{iteration}"), baseline)
            .unwrap();
        let root = provider.workspace_path(handle.id()).unwrap();
        assert_eq!(
            fs::read(root.join(&args.probe_path)).unwrap(),
            probe_expected
        );
        visible_ms.push(visible_started.elapsed().as_secs_f64() * 1_000.0);
        end_to_end_ms.push(end_to_end_started.elapsed().as_secs_f64() * 1_000.0);
        core.remove_workspace(handle).unwrap();
    }

    let parallel_started = Instant::now();
    let parallel_baseline = capture_repository(&args.repository, core.chunks()).unwrap();
    let parallel_owner = core
        .create_workspace("perf-parallel-owner", parallel_baseline.clone())
        .unwrap();
    let owner_root = provider.workspace_path(parallel_owner.id()).unwrap();
    assert_eq!(
        fs::read(owner_root.join(&args.probe_path)).unwrap(),
        probe_expected
    );
    let data_root = args.data_root.clone();
    let probe_path = args.probe_path.clone();
    let probe_expected_parallel = probe_expected.clone();
    let workers = (1..args.parallel)
        .map(|index| {
            let data_root = data_root.clone();
            let probe_path = probe_path.clone();
            let expected = probe_expected_parallel.clone();
            let baseline = parallel_baseline.clone();
            thread::spawn(move || {
                let core = WorkspaceCore::open(data_root.join("core")).unwrap();
                let provider = ProviderInstallation::require_healthy(&data_root).unwrap();
                let handle = core
                    .create_workspace_from_shared_baseline(
                        &format!("perf-parallel-{index}"),
                        &baseline,
                    )
                    .unwrap();
                let root = provider.workspace_path(handle.id()).unwrap();
                assert_eq!(fs::read(root.join(probe_path)).unwrap(), expected);
                handle
            })
        })
        .collect::<Vec<_>>();
    let mut handles = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .collect::<Vec<_>>();
    handles.push(parallel_owner);
    let parallel_ms = parallel_started.elapsed().as_secs_f64() * 1_000.0;
    for handle in handles {
        core.remove_workspace(handle).unwrap();
    }

    let (native_worktree_ms, native_worktree_physical_bytes) = measure_native_worktrees(
        &args.repository,
        &args.native_baseline_root,
        args.native_baseline_iterations,
    );

    let large_path = args.repository.join(".greppy-perf-large.bin");
    if large_path.exists() {
        fail("fixture unexpectedly contains .greppy-perf-large.bin");
    }
    fs::write(&large_path, vec![7_u8; CHUNK_SIZE * 3]).unwrap();
    let write_baseline = capture_repository(&args.repository, core.chunks()).unwrap();
    let write_handle = core
        .create_workspace("perf-one-byte", write_baseline)
        .unwrap();
    let write_root = provider.workspace_path(write_handle.id()).unwrap();
    let chunks_before = core.chunks().stats().unwrap();
    let bytes_before = allocated_tree_bytes(&args.data_root);
    let mut large = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(write_root.join(".greppy-perf-large.bin"))
        .unwrap();
    large.seek(SeekFrom::Start(CHUNK_SIZE as u64 + 17)).unwrap();
    large.write_all(&[9]).unwrap();
    large.sync_all().unwrap();
    drop(large);
    let chunks_after = core.chunks().stats().unwrap();
    let bytes_after = allocated_tree_bytes(&args.data_root);
    let one_byte_physical_delta = bytes_after.saturating_sub(bytes_before);
    let one_byte_chunk_delta = chunks_after
        .chunk_count
        .saturating_sub(chunks_before.chunk_count);
    core.remove_workspace(write_handle).unwrap();
    fs::remove_file(&large_path).unwrap();

    let toolchains = args
        .toolchain_cases
        .iter()
        .map(|raw| {
            serde_json::from_str::<ToolchainCase>(raw)
                .unwrap_or_else(|error| fail(&format!("invalid --toolchain-case JSON: {error}")))
        })
        .map(|case| measure_toolchain(&core, &provider, &args.repository, case))
        .collect::<Vec<_>>();

    let visible_p50_ms = percentile(&visible_ms, 50);
    let visible_p95_ms = percentile(&visible_ms, 95);
    let end_to_end_p50_ms = percentile(&end_to_end_ms, 50);
    let end_to_end_p95_ms = percentile(&end_to_end_ms, 95);
    let snapshot_p50_ms = percentile(&snapshot_ms, 50);
    let snapshot_p95_ms = percentile(&snapshot_ms, 95);
    let provider_sha256 = sha256_file(&args.provider_binary);
    let fixture_commit = git(&args.repository, &["rev-parse", "HEAD"]);
    let native_worktree_p50_ms = percentile(&native_worktree_ms, 50);
    let native_worktree_p95_ms = percentile(&native_worktree_ms, 95);
    let native_worktree_physical_p50_bytes = percentile_u64(&native_worktree_physical_bytes, 50);
    let creation_improvement_percent = if native_worktree_p95_ms == 0.0 {
        0.0
    } else {
        (native_worktree_p95_ms - end_to_end_p95_ms) * 100.0 / native_worktree_p95_ms
    };
    let untouched_space_reduction_percent = if native_worktree_physical_p50_bytes == 0 {
        0.0
    } else {
        (native_worktree_physical_p50_bytes as f64 - untouched_physical_delta as f64) * 100.0
            / native_worktree_physical_p50_bytes as f64
    };
    let evidence = serde_json::json!({
        "schema": "greppy.portable-cow-performance.v1",
        "source_commit": args.source_commit,
        "source_tracked_worktree_dirty": source_tree_dirty(),
        "provider_binary": args.provider_binary,
        "provider_sha256": provider_sha256,
        "profile": "release",
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "hardware": args.hardware,
        "fixture_repository": args.repository,
        "fixture_commit": fixture_commit,
        "fixture_tracked_files": tracked_files,
        "iterations": args.iterations,
        "workspace_creation": {
            "snapshot_p50_ms": snapshot_p50_ms,
            "snapshot_p95_ms": snapshot_p95_ms,
            "visible_p50_ms": visible_p50_ms,
            "visible_p95_ms": visible_p95_ms,
            "end_to_end_p50_ms": end_to_end_p50_ms,
            "end_to_end_p95_ms": end_to_end_p95_ms,
            "end_to_end_p95_gate_ms": args.max_visible_p95_ms,
        },
        "native_git_worktree_baseline": {
            "description": "warm git worktree add --detach checkout on the identical repository and host",
            "iterations": args.native_baseline_iterations,
            "p50_ms": native_worktree_p50_ms,
            "p95_ms": native_worktree_p95_ms,
            "physical_p50_bytes": native_worktree_physical_p50_bytes,
        },
        "comparison": {
            "portable_creation_improvement_percent_vs_git_worktree": creation_improvement_percent,
            "portable_untouched_space_reduction_percent_vs_git_worktree": untouched_space_reduction_percent,
        },
        "space": {
            "untouched_physical_delta_bytes": untouched_physical_delta,
            "untouched_gate_bytes": args.max_untouched_bytes,
            "one_byte_write_physical_delta_bytes": one_byte_physical_delta,
            "one_byte_write_gate_bytes": args.max_one_byte_write_bytes,
            "one_byte_write_new_chunks": one_byte_chunk_delta,
            "chunk_size": CHUNK_SIZE,
        },
        "parallel": {
            "workspaces": args.parallel,
            "wall_ms": parallel_ms,
        },
        "toolchains": toolchains,
    });
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let encoded = serde_json::to_vec_pretty(&evidence).unwrap();
    fs::write(&args.output, &encoded).unwrap();
    println!("{}", String::from_utf8(encoded).unwrap());

    if args.enforce {
        if source_tree_dirty() {
            fail("release evidence source worktree is tracked-dirty");
        }
        if end_to_end_p95_ms > args.max_visible_p95_ms {
            fail(&format!(
                "end-to-end workspace P95 {end_to_end_p95_ms:.3} ms exceeds gate"
            ));
        }
        if untouched_physical_delta > args.max_untouched_bytes {
            fail("untouched workspace physical delta exceeds gate");
        }
        if one_byte_chunk_delta != 1 || one_byte_physical_delta > args.max_one_byte_write_bytes {
            fail("one-byte write violates Chunk-CoW physical-delta gate");
        }
        if args.parallel != 50 {
            fail("release evidence must exercise exactly 50 parallel workspaces");
        }
        for required in ["rust", "python", "node"] {
            if !toolchains.iter().any(|case| case.name == required) {
                fail(&format!("missing required {required} toolchain case"));
            }
        }
        if toolchains.iter().any(|case| case.overhead_percent > 20.0) {
            fail("toolchain overhead exceeds 20% gate");
        }
    }
}

fn measure_toolchain(
    core: &WorkspaceCore,
    provider: &ProviderInstallation,
    repository: &Path,
    case: ToolchainCase,
) -> ToolchainResult {
    if case.argv.is_empty() {
        fail("toolchain argv must not be empty");
    }
    let baseline = capture_repository(repository, core.chunks()).unwrap();
    let handle = core
        .create_workspace(&format!("perf-toolchain-{}", case.name), baseline)
        .unwrap();
    let workspace = provider.workspace_path(handle.id()).unwrap();
    let native_cwd = repository.join(&case.cwd);
    let workspace_cwd = workspace.join(&case.cwd);
    run_argv(&native_cwd, &case.argv);
    run_argv(&workspace_cwd, &case.argv);
    let mut native_samples = Vec::with_capacity(5);
    let mut workspace_samples = Vec::with_capacity(5);
    for round in 0..5 {
        if round % 2 == 0 {
            native_samples.push(timed_argv(&native_cwd, &case.argv).as_secs_f64() * 1_000.0);
            workspace_samples.push(timed_argv(&workspace_cwd, &case.argv).as_secs_f64() * 1_000.0);
        } else {
            workspace_samples.push(timed_argv(&workspace_cwd, &case.argv).as_secs_f64() * 1_000.0);
            native_samples.push(timed_argv(&native_cwd, &case.argv).as_secs_f64() * 1_000.0);
        }
    }
    core.remove_workspace(handle).unwrap();
    let native_ms = percentile(&native_samples, 50);
    let workspace_ms = percentile(&workspace_samples, 50);
    let overhead = if native_ms == 0.0 {
        0.0
    } else {
        (workspace_ms - native_ms) * 100.0 / native_ms
    };
    ToolchainResult {
        name: case.name,
        native_ms,
        workspace_ms,
        overhead_percent: overhead,
    }
}

fn measure_native_worktrees(
    repository: &Path,
    root: &Path,
    iterations: usize,
) -> (Vec<f64>, Vec<u64>) {
    if iterations < 3 {
        fail("at least three native Git-worktree baseline iterations are required");
    }
    if root.exists() {
        fail("native baseline root must not exist before the measurement");
    }
    fs::create_dir_all(root).unwrap();
    let git_worktrees = PathBuf::from(git(
        repository,
        &[
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            "worktrees",
        ],
    ));
    let mut elapsed = Vec::with_capacity(iterations);
    let mut physical = Vec::with_capacity(iterations);
    for iteration in 0..iterations {
        let target = root.join(format!("native-{iteration}"));
        let metadata_before = allocated_tree_bytes_if_exists(&git_worktrees);
        let started = Instant::now();
        let output = Command::new("git")
            .args(["worktree", "add", "--quiet", "--detach"])
            .arg(&target)
            .arg("HEAD")
            .current_dir(repository)
            .output()
            .unwrap();
        if !output.status.success() {
            fail(&format!(
                "native git worktree baseline failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        elapsed.push(started.elapsed().as_secs_f64() * 1_000.0);
        let metadata_after = allocated_tree_bytes_if_exists(&git_worktrees);
        physical.push(
            allocated_tree_bytes(&target)
                .saturating_add(metadata_after.saturating_sub(metadata_before)),
        );
        let output = Command::new("git")
            .args(["worktree", "remove", "--force"])
            .arg(&target)
            .current_dir(repository)
            .output()
            .unwrap();
        if !output.status.success() {
            fail(&format!(
                "native git worktree cleanup failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
    }
    fs::remove_dir(root).unwrap();
    (elapsed, physical)
}

fn run_argv(cwd: &Path, argv: &[String]) {
    let status = Command::new(&argv[0])
        .args(&argv[1..])
        .current_dir(cwd)
        .status()
        .unwrap_or_else(|error| fail(&format!("cannot run {argv:?}: {error}")));
    if !status.success() {
        fail(&format!("toolchain command {argv:?} failed with {status}"));
    }
}

fn timed_argv(cwd: &Path, argv: &[String]) -> Duration {
    let started = Instant::now();
    run_argv(cwd, argv);
    started.elapsed()
}

fn tracked_file_count(repository: &Path) -> usize {
    let bytes = git_bytes(repository, &["ls-files", "-z"]);
    bytes.iter().filter(|byte| **byte == 0).count()
}

fn percentile(samples: &[f64], percentile: usize) -> f64 {
    let mut values = samples.to_vec();
    values.sort_by(f64::total_cmp);
    let rank = (values.len() * percentile).div_ceil(100).saturating_sub(1);
    values[rank]
}

fn percentile_u64(samples: &[u64], percentile: usize) -> u64 {
    let mut values = samples.to_vec();
    values.sort_unstable();
    let rank = (values.len() * percentile).div_ceil(100).saturating_sub(1);
    values[rank]
}

fn source_tree_dirty() -> bool {
    let source_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    !git(
        &source_root,
        &["status", "--porcelain", "--untracked-files=no"],
    )
    .is_empty()
}

fn git(repository: &Path, args: &[&str]) -> String {
    String::from_utf8(git_bytes(repository, args))
        .unwrap_or_else(|_| fail("Git output is not UTF-8"))
        .trim()
        .into()
}

fn git_bytes(repository: &Path, args: &[&str]) -> Vec<u8> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repository)
        .output()
        .unwrap_or_else(|error| fail(&format!("cannot execute Git: {error}")));
    if !output.status.success() {
        fail(&format!(
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    output.stdout
}

fn sha256_file(path: &Path) -> String {
    let bytes =
        fs::read(path).unwrap_or_else(|error| fail(&format!("cannot hash provider: {error}")));
    format!("{:x}", Sha256::digest(bytes))
}

fn allocated_tree_bytes(root: &Path) -> u64 {
    let metadata = fs::symlink_metadata(root).unwrap();
    let mut total = allocated_bytes(root, &metadata);
    if metadata.is_dir() && !metadata.file_type().is_symlink() {
        for entry in fs::read_dir(root).unwrap() {
            total = total.saturating_add(allocated_tree_bytes(&entry.unwrap().path()));
        }
    }
    total
}

fn allocated_tree_bytes_if_exists(root: &Path) -> u64 {
    if root.exists() {
        allocated_tree_bytes(root)
    } else {
        0
    }
}

#[cfg(unix)]
fn allocated_bytes(_path: &Path, metadata: &fs::Metadata) -> u64 {
    use std::os::unix::fs::MetadataExt as _;
    metadata.blocks().saturating_mul(512)
}

#[cfg(windows)]
fn allocated_bytes(path: &Path, metadata: &fs::Metadata) -> u64 {
    use std::os::windows::ffi::OsStrExt as _;
    use windows_sys::Win32::Foundation::{GetLastError, INVALID_FILE_SIZE};
    use windows_sys::Win32::Storage::FileSystem::GetCompressedFileSizeW;
    if metadata.is_dir() {
        return 0;
    }
    let wide = path
        .as_os_str()
        .encode_wide()
        .chain([0])
        .collect::<Vec<_>>();
    let mut high = 0_u32;
    let low = unsafe { GetCompressedFileSizeW(wide.as_ptr(), &mut high) };
    if low == INVALID_FILE_SIZE && unsafe { GetLastError() } != 0 {
        fail(&format!(
            "cannot measure allocated bytes for {}",
            path.display()
        ));
    }
    (u64::from(high) << 32) | u64::from(low)
}

fn fail(message: &str) -> ! {
    eprintln!("portable workspace performance gate failed: {message}");
    std::process::exit(1)
}
