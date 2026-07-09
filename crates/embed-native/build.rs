use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(embed_native_has_metallib)");
    println!("cargo:rustc-check-cfg=cfg(embed_native_has_tensor_metallib)");
    println!("cargo:rustc-check-cfg=cfg(embed_native_has_cuda_dylib)");
    println!("cargo:rerun-if-env-changed=EMBED_NATIVE_METAL_MIN_OS");
    println!("cargo:rerun-if-env-changed=EMBED_NATIVE_METAL_BASE_STD");
    println!("cargo:rerun-if-env-changed=EMBED_NATIVE_METAL_TENSOR_STD");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH_LIST");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=NVCC");

    if env::var_os("CARGO_FEATURE_CUDA").is_some() {
        build_cuda();
    }

    if env::var_os("CARGO_FEATURE_METAL").is_some() {
        if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
            println!(
                "cargo:warning=greppy-embed-native: `metal` feature is only supported on macOS"
            );
            return;
        }
        build_macos_metal();
    }
}

fn build_cuda() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "linux" && target_os != "windows" {
        println!(
            "cargo:warning=greppy-embed-native: `cuda` feature is only supported on Linux/Windows"
        );
        return;
    }

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let cuda_dir = manifest.join("vendor/cuda");
    let ggml_cuda = cuda_dir.join("ggml-cuda");
    let ggml_include = cuda_dir.join("ggml-include");
    let wrapper = cuda_dir.join("embed_native_cuda.cu");
    let quantize = ggml_cuda.join("quantize.cu");
    let mmvq = ggml_cuda.join("mmvq.cu");
    let mmvq_h = ggml_cuda.join("mmvq.cuh");
    let unary_h = ggml_cuda.join("unary.cuh");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    println!("cargo:rerun-if-changed={}", wrapper.display());
    println!("cargo:rerun-if-changed={}", quantize.display());
    println!("cargo:rerun-if-changed={}", mmvq.display());
    println!("cargo:rerun-if-changed={}", mmvq_h.display());
    println!("cargo:rerun-if-changed={}", unary_h.display());
    println!("cargo:rerun-if-changed={}", ggml_cuda.display());
    println!("cargo:rerun-if-changed={}", ggml_include.display());

    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".into());
    let arch_list = cuda_arch_list();
    let ext = if target_os == "windows" { "dll" } else { "so" };
    let lib = out_dir.join(format!("greppy_embed_native_cuda.{ext}"));
    let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| {
        if target_os == "windows" {
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6".into()
        } else {
            "/usr/local/cuda".into()
        }
    });
    compile_cuda_dylib(
        &nvcc,
        &arch_list,
        &target_os,
        &cuda_home,
        &wrapper,
        &quantize,
        &lib,
        &ggml_cuda,
        &ggml_include,
        &cuda_dir,
    );
    println!(
        "cargo:rustc-env=GREPPY_EMBED_NATIVE_CUDA_DYLIB={}",
        lib.display()
    );
    println!("cargo:rustc-cfg=embed_native_has_cuda_dylib");
}

fn cuda_arch_list() -> Vec<String> {
    if let Ok(value) = env::var("CUDA_ARCH_LIST").or_else(|_| env::var("CUDA_COMPUTE_CAP")) {
        let archs = value
            .split(|c| c == ',' || c == ';' || c == ' ')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.trim_start_matches("sm_").replace('.', ""))
            .collect::<Vec<_>>();
        if !archs.is_empty() {
            return archs;
        }
    }
    vec![
        "75".into(),
        "80".into(),
        "86".into(),
        "89".into(),
        "90".into(),
    ]
}

fn compile_cuda_dylib(
    nvcc: &str,
    arch_list: &[String],
    target_os: &str,
    cuda_home: &str,
    wrapper: &Path,
    quantize: &Path,
    lib: &Path,
    ggml_cuda: &Path,
    ggml_include: &Path,
    cuda_dir: &Path,
) {
    let mut cmd = Command::new(nvcc);
    cmd.args(["-shared", "-std=c++17", "-O3", "--expt-relaxed-constexpr"]);
    if target_os != "windows" {
        cmd.args(["-Xcompiler", "-fPIC"]);
    }
    for arch in arch_list {
        cmd.arg("-gencode")
            .arg(format!("arch=compute_{arch},code=sm_{arch}"));
    }
    if let Some(last) = arch_list.last() {
        cmd.arg("-gencode")
            .arg(format!("arch=compute_{last},code=compute_{last}"));
    }
    if target_os == "windows" {
        // windows-latest ships a newer MSVC than the CUDA toolkit officially
        // pins; nvcc's host_config.h otherwise hard-errors ("unsupported
        // Microsoft Visual Studio version"). Our kernels don't touch the
        // version-sensitive surface, so allow the newer host compiler.
        cmd.arg("-allow-unsupported-compiler");
        let cuda_include = PathBuf::from(cuda_home).join("include");
        if cuda_include.is_dir() {
            cmd.arg(format!("-I{}", cuda_include.display()));
        }
        let cuda_lib = PathBuf::from(cuda_home).join("lib").join("x64");
        if cuda_lib.is_dir() {
            cmd.arg(format!("-L{}", cuda_lib.display()));
        }
    } else {
        let cuda_home = PathBuf::from(cuda_home);
        let cuda_include = cuda_home.join("include");
        let cuda_target = cuda_home.join("targets").join("x86_64-linux");
        let cuda_target_include = cuda_target.join("include");
        if cuda_include.is_dir() {
            cmd.arg(format!("-I{}", cuda_include.display()));
        }
        if cuda_target_include.is_dir() {
            cmd.arg(format!("-I{}", cuda_target_include.display()));
        }
        let cuda_lib64 = cuda_home.join("lib64");
        let cuda_stubs = cuda_lib64.join("stubs");
        let cuda_target_lib = cuda_target.join("lib");
        let cuda_target_stubs = cuda_target_lib.join("stubs");
        if cuda_lib64.is_dir() {
            cmd.arg(format!("-L{}", cuda_lib64.display()));
            cmd.args(["-Xlinker", "-rpath", "-Xlinker"]).arg(cuda_lib64);
        }
        if cuda_stubs.is_dir() {
            cmd.arg(format!("-L{}", cuda_stubs.display()));
        }
        if cuda_target_lib.is_dir() {
            cmd.arg(format!("-L{}", cuda_target_lib.display()));
            cmd.args(["-Xlinker", "-rpath", "-Xlinker"])
                .arg(cuda_target_lib);
        }
        if cuda_target_stubs.is_dir() {
            cmd.arg(format!("-L{}", cuda_target_stubs.display()));
        }
    }
    cmd.args(["-DGGML_CUDA_FORCE_MMQ", "-I"])
        .arg(ggml_cuda)
        .arg("-I")
        .arg(ggml_include)
        .arg("-I")
        .arg(cuda_dir)
        .arg("-o")
        .arg(lib)
        .arg(wrapper)
        .arg(quantize)
        .arg("-lcublas")
        .arg("-lcuda");

    match cmd.output() {
        Ok(output) if output.status.success() => {}
        Ok(output) => {
            for line in String::from_utf8_lossy(&output.stderr).lines() {
                println!("cargo:warning=greppy-embed-native: nvcc: {line}");
            }
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                println!("cargo:warning=greppy-embed-native: nvcc: {line}");
            }
            panic!(
                "greppy-embed-native: nvcc failed while building CUDA dylib {} (exit {:?})",
                lib.display(),
                output.status
            );
        }
        Err(err) => {
            panic!("greppy-embed-native: nvcc unavailable for CUDA build ({err})");
        }
    }
}

fn build_macos_metal() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let shader_dir = manifest.join("vendor/metal/shaders/ggml");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    println!("cargo:rerun-if-changed={}", shader_dir.display());
    let shader_files = collect_metal_shaders(&shader_dir);
    if shader_files.is_empty() {
        println!(
            "cargo:warning=greppy-embed-native: no .metal shaders under {}; Metal runtime will not load",
            shader_dir.display()
        );
        return;
    }

    let min_os = env::var("EMBED_NATIVE_METAL_MIN_OS").unwrap_or_else(|_| "14.0".into());
    let base_std =
        env::var("EMBED_NATIVE_METAL_BASE_STD").unwrap_or_else(|_| "-std=metal3.1".into());
    let tensor_std =
        env::var("EMBED_NATIVE_METAL_TENSOR_STD").unwrap_or_else(|_| "-std=metal4.0".into());

    let base_metallib = compile_metallib(
        &manifest,
        &shader_dir,
        &out_dir,
        &shader_files,
        "base",
        &base_std,
        &min_os,
        false,
        true,
    )
    .expect("greppy-embed-native: required base Metal metallib was not produced");
    let tensor_metallib = compile_metallib(
        &manifest,
        &shader_dir,
        &out_dir,
        &shader_files,
        "tensor",
        &tensor_std,
        &min_os,
        true,
        false,
    );
    println!(
        "cargo:rustc-env=GREPPY_EMBED_NATIVE_METALLIB_BASE={}",
        base_metallib.display()
    );
    if let Some(tensor_metallib) = tensor_metallib {
        println!(
            "cargo:rustc-env=GREPPY_EMBED_NATIVE_METALLIB_TENSOR={}",
            tensor_metallib.display()
        );
        println!("cargo:rustc-cfg=embed_native_has_tensor_metallib");
    }
    println!("cargo:rustc-cfg=embed_native_has_metallib");
}

fn compile_metallib(
    manifest: &Path,
    shader_dir: &Path,
    out_dir: &Path,
    shader_files: &[PathBuf],
    flavor: &str,
    std_flag: &str,
    min_os: &str,
    tensor: bool,
    required: bool,
) -> Option<PathBuf> {
    let mut air_files = Vec::with_capacity(shader_files.len());

    for src in shader_files {
        let rel = src.strip_prefix(manifest).unwrap_or(src);
        let munged = rel
            .with_extension("")
            .to_string_lossy()
            .replace(std::path::MAIN_SEPARATOR, "_");
        let air = out_dir.join(format!("{flavor}_{munged}.air"));
        println!("cargo:rerun-if-changed={}", src.display());

        let mut cmd = Command::new("xcrun");
        cmd.args(["-sdk", "macosx", "metal", "-c"])
            .arg(format!("-I{}", shader_dir.display()))
            .args([
                "-Wall",
                "-Wextra",
                "-fno-fast-math",
                "-Wno-c++17-extensions",
                "-Wno-c++20-extensions",
                &format!("-mmacosx-version-min={min_os}"),
                "-O3",
            ]);
        if tensor {
            cmd.arg("-DGGML_METAL_HAS_BF16");
            cmd.arg("-DGGML_METAL_HAS_TENSOR");
        }
        if !std_flag.trim().is_empty() {
            cmd.arg(std_flag);
        }
        cmd.arg("-o").arg(&air).arg(src);

        match cmd.output() {
            Ok(output) if output.status.success() => air_files.push(air),
            Ok(output) => {
                for line in String::from_utf8_lossy(&output.stderr).lines() {
                    println!("cargo:warning=greppy-embed-native: xcrun metal: {line}");
                }
                for line in String::from_utf8_lossy(&output.stdout).lines() {
                    println!("cargo:warning=greppy-embed-native: xcrun metal: {line}");
                }
                let msg = format!(
                    "greppy-embed-native: xcrun metal failed for {} ({flavor}, exit {:?})",
                    src.display(),
                    output.status
                );
                if required {
                    panic!("{msg}");
                }
                println!(
                    "cargo:warning={msg}; optional {flavor} metallib skipped, using base simdgroup Metal library"
                );
                return None;
            }
            Err(err) => {
                let msg = format!("greppy-embed-native: xcrun metal unavailable ({err})");
                if required {
                    panic!("{msg}");
                }
                println!(
                    "cargo:warning={msg}; optional {flavor} metallib skipped, using base simdgroup Metal library"
                );
                return None;
            }
        }
    }

    let metallib = out_dir.join(format!("greppy_embed_native_{flavor}.metallib"));
    match Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib", "-o"])
        .arg(&metallib)
        .args(&air_files)
        .output()
    {
        Ok(output) if output.status.success() => Some(metallib),
        Ok(output) => {
            for line in String::from_utf8_lossy(&output.stderr).lines() {
                println!("cargo:warning=greppy-embed-native: xcrun metallib: {line}");
            }
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                println!("cargo:warning=greppy-embed-native: xcrun metallib: {line}");
            }
            let msg = format!(
                "greppy-embed-native: xcrun metallib failed for {flavor} (exit {:?})",
                output.status
            );
            if required {
                panic!("{msg}");
            }
            println!(
                "cargo:warning={msg}; optional {flavor} metallib skipped, using base simdgroup Metal library"
            );
            None
        }
        Err(err) => {
            let msg = format!("greppy-embed-native: xcrun metallib unavailable ({err})");
            if required {
                panic!("{msg}");
            }
            println!(
                "cargo:warning={msg}; optional {flavor} metallib skipped, using base simdgroup Metal library"
            );
            None
        }
    }
}

fn collect_metal_shaders(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(root) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(collect_metal_shaders(&path));
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("metal") {
            out.push(path);
        }
    }
    out.sort();
    out
}
