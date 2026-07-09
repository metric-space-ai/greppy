//! Metal FFI layer for the Apple-Silicon backend.
//!
//! Thin Rust wrapper over `MTLDevice` / `MTLCommandQueue` /
//! `MTLLibrary` / `MTLComputePipelineState` / `MTLBuffer`, with the
//! crate's precompiled metallib (built by `build.rs`) loaded on
//! device creation. Higher-level code (`src/metal/kernels.rs`,
//! `src/metal/qwen.rs`, etc.) only ever sees the types and helpers
//! defined here — no `msg_send!` or `objc2::runtime::*` leaks upwards.
//!
//! # Lifetime model
//!
//! `Device` owns the root `MTLDevice` and the loaded `MTLLibrary`. A
//! `Device` is `Send + Sync` because `MTLDevice`/`MTLLibrary` are
//! thread-safe per Apple's doc. `PipelineCache` memoizes
//! `MTLComputePipelineState` lookups by kernel name and is parked on
//! the `Device`.
//!
//! Buffers (`Buffer`) are reference-counted through
//! `objc2::rc::Retained`. `CommandBuffer` is one-shot — you create it
//! per dispatch batch via `Device::new_command_buffer()`, encode
//! compute passes through `ComputeEncoder`, then commit + wait.
//!
//! # Metallib provenance
//!
//! `build.rs` sets the env var `GREPPY_EMBED_NATIVE_METALLIB` to the absolute
//! path of the compiled metallib in `$OUT_DIR`. We pull it into the
//! binary at compile time via `include_bytes!` so the installed CTOX
//! doesn't need a companion file on disk — the shaders ship with the
//! executable.

use std::ffi::{c_void, CStr};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, NSUInteger};
use objc2_metal::{
    MTLBarrierScope, MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLDispatchType, MTLGPUFamily, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

use crate::metal::errors::set_last_error;

static DISPATCH_COUNT: AtomicU64 = AtomicU64::new(0);
static BUFFER_ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static BUFFER_ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static METALLIB_LOAD_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, Default)]
pub struct BufferAllocStats {
    pub count: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CommandBufferTiming {
    pub submit_wait_secs: f64,
    pub kernel_secs: f64,
    pub gpu_secs: f64,
}

pub fn dispatch_count_snapshot() -> u64 {
    DISPATCH_COUNT.load(Ordering::Relaxed)
}

pub fn buffer_alloc_stats_snapshot() -> BufferAllocStats {
    BufferAllocStats {
        count: BUFFER_ALLOC_COUNT.load(Ordering::Relaxed),
        bytes: BUFFER_ALLOC_BYTES.load(Ordering::Relaxed),
    }
}

// ─── Metallib blob ─────────────────────────────────────────────────
//
// The path is produced by build.rs (see `build_macos_metal`). When
// it's absent (e.g. `cargo check` on a system without `xcrun metal`),
// we fall back to an empty slice and every pipeline lookup will fail
// with a clear error message rather than linking to nothing.

#[cfg(all(target_os = "macos", embed_native_has_metallib))]
const METALLIB_BASE_BLOB: &[u8] = include_bytes!(env!("GREPPY_EMBED_NATIVE_METALLIB_BASE"));

#[cfg(all(target_os = "macos", not(embed_native_has_metallib)))]
const METALLIB_BASE_BLOB: &[u8] = &[];

#[cfg(all(target_os = "macos", embed_native_has_tensor_metallib))]
const METALLIB_TENSOR_BLOB: &[u8] = include_bytes!(env!("GREPPY_EMBED_NATIVE_METALLIB_TENSOR"));

#[cfg(all(target_os = "macos", not(embed_native_has_tensor_metallib)))]
const METALLIB_TENSOR_BLOB: &[u8] = &[];

// build.rs sets `cargo:rustc-cfg=embed_native_has_metallib` iff the metallib
// actually got produced.  See the tail of build_macos_metal.

// ─── Device ────────────────────────────────────────────────────────

/// Root handle into the Metal device + its loaded kernel library.
///
/// A single `Device` is shared across the entire model — there is only
/// one Apple Silicon GPU in the system, and creating it is not cheap.
pub struct Device {
    mtl: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    base_library: Option<Retained<ProtocolObject<dyn MTLLibrary>>>,
    tensor_library: Option<Retained<ProtocolObject<dyn MTLLibrary>>>,
    supports_metal4: bool,
    pipelines: Mutex<
        Vec<(
            String,
            Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        )>,
    >,
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    /// Grab the system default MTLDevice and load the precompiled
    /// metallib blob. Returns `None` if Metal is unavailable (which
    /// can happen on some CI hosts or VMs without GPU passthrough).
    pub fn default_system() -> Option<Self> {
        let raw: *mut ProtocolObject<dyn MTLDevice> = unsafe { MTLCreateSystemDefaultDevice() };
        if raw.is_null() {
            return None;
        }
        let mtl: Retained<ProtocolObject<dyn MTLDevice>> = unsafe { Retained::from_raw(raw) }?;
        let queue = mtl.newCommandQueue()?;

        let base_library = if METALLIB_BASE_BLOB.is_empty() {
            set_last_error(
                "greppy-embed-native: metallib blob empty — build.rs \
                 produced no base `greppy_embed_native_base.metallib`. \
                 Runtime dispatch will fail for every kernel.",
            );
            None
        } else {
            load_library_from_blob(&mtl, METALLIB_BASE_BLOB, "base")
        };

        let supports_metal4 = mtl.supportsFamily(MTLGPUFamily(5002));
        let tensor_library = maybe_load_tensor_library(&mtl, supports_metal4);

        Some(Self {
            mtl,
            queue,
            base_library,
            tensor_library,
            supports_metal4,
            pipelines: Mutex::new(Vec::new()),
        })
    }

    /// Cached pipeline for `kernel_name`. First lookup does a
    /// `newFunctionWithName:` + `newComputePipelineStateWithFunction:`
    /// roundtrip; subsequent lookups are O(n) over the cache (small,
    /// single-digits in practice).
    ///
    /// Works for kernels that either (a) don't declare any
    /// function_constants, or (b) declare function_constants with
    /// default values that don't need to be overridden. For kernels
    /// that require explicit constant bindings, use
    /// [`Self::pipeline_with_constants`].
    pub fn pipeline(
        &self,
        kernel_name: &str,
    ) -> Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        let (library, flavor) = self.library_for_kernel(kernel_name)?;
        let cache_key = format!("{flavor}:{kernel_name}");
        {
            let guard = self.pipelines.lock().ok()?;
            for (name, pso) in guard.iter() {
                if name == &cache_key {
                    return Some(pso.clone());
                }
            }
        }

        let ns_name = NSString::from_str(kernel_name);
        let func = library.newFunctionWithName(&ns_name)?;
        let pso = self
            .mtl
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| {
                set_last_error(format!(
                    "newComputePipelineStateWithFunction({kernel_name}) failed: {e:?}"
                ));
            })
            .ok()?;

        self.pipelines.lock().ok()?.push((cache_key, pso.clone()));
        Some(pso)
    }

    /// Pipeline with explicit function-constant bindings. Mirrors
    /// `ggml_metal_library_compile_pipeline(lib, base, name, cv)` in
    /// llama.cpp's `ggml-metal-device.m`. The cache key is
    /// `cache_key` — callers should pass something like
    /// `"kernel_rope_multi_f32#imrope=true"` so two invocations with
    /// different FC values don't collide.
    pub fn pipeline_with_constants(
        &self,
        cache_key: &str,
        kernel_name: &str,
        setup: impl FnOnce(&objc2_metal::MTLFunctionConstantValues),
    ) -> Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        let (library, flavor) = self.library_for_kernel(kernel_name)?;
        let cache_key = format!("{flavor}:{cache_key}");
        {
            let guard = self.pipelines.lock().ok()?;
            for (name, pso) in guard.iter() {
                if name == &cache_key {
                    return Some(pso.clone());
                }
            }
        }

        let cv = objc2_metal::MTLFunctionConstantValues::new();
        setup(&cv);
        let ns_name = NSString::from_str(kernel_name);
        let func = library
            .newFunctionWithName_constantValues_error(&ns_name, &cv)
            .map_err(|e| {
                let desc = unsafe { e.localizedDescription() };
                let s: &NSString = desc.as_ref();
                set_last_error(format!(
                    "newFunctionWithName_constantValues({kernel_name}) failed: {}",
                    s
                ));
            })
            .ok()?;
        let pso = self
            .mtl
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| {
                set_last_error(format!(
                    "newComputePipelineStateWithFunction({cache_key}) failed: {e:?}"
                ));
            })
            .ok()?;

        self.pipelines
            .lock()
            .ok()?
            .push((cache_key.to_string(), pso.clone()));
        Some(pso)
    }

    pub fn uses_tensor_mul_mm(&self) -> bool {
        self.tensor_library.is_some()
    }

    pub fn supports_metal4(&self) -> bool {
        self.supports_metal4
    }

    fn library_for_kernel(
        &self,
        kernel_name: &str,
    ) -> Option<(&Retained<ProtocolObject<dyn MTLLibrary>>, &'static str)> {
        if is_tensor_mul_mm_kernel(kernel_name) {
            if let Some(lib) = self.tensor_library.as_ref() {
                return Some((lib, "tensor"));
            }
        }
        self.base_library.as_ref().map(|lib| (lib, "base"))
    }

    /// Allocate a new shared (CPU+GPU accessible) buffer of
    /// `byte_len` bytes. Bytes are zero-initialized by Metal.
    pub fn new_buffer(&self, byte_len: usize) -> Option<Buffer> {
        let opts = MTLResourceOptions::MTLResourceStorageModeShared;
        let buf = self
            .mtl
            .newBufferWithLength_options(byte_len as NSUInteger, opts)?;
        BUFFER_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        BUFFER_ALLOC_BYTES.fetch_add(byte_len as u64, Ordering::Relaxed);
        Some(Buffer { inner: buf })
    }

    /// Allocate a GPU-only buffer for intermediates that are never read or
    /// written by the CPU. The final output is copied to a shared readback
    /// buffer before host access.
    pub fn new_private_buffer(&self, byte_len: usize) -> Option<Buffer> {
        let opts = MTLResourceOptions::MTLResourceStorageModePrivate;
        let buf = self
            .mtl
            .newBufferWithLength_options(byte_len as NSUInteger, opts)?;
        BUFFER_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        BUFFER_ALLOC_BYTES.fetch_add(byte_len as u64, Ordering::Relaxed);
        Some(Buffer { inner: buf })
    }

    /// Allocate a shared buffer and memcpy `src` into it.
    pub fn new_buffer_from_slice(&self, src: &[u8]) -> Option<Buffer> {
        let opts = MTLResourceOptions::MTLResourceStorageModeShared;
        let buf = unsafe {
            self.mtl.newBufferWithBytes_length_options(
                std::ptr::NonNull::new(src.as_ptr() as *mut c_void)?,
                src.len() as NSUInteger,
                opts,
            )
        }?;
        BUFFER_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        BUFFER_ALLOC_BYTES.fetch_add(src.len() as u64, Ordering::Relaxed);
        Some(Buffer { inner: buf })
    }

    /// Open a new `CommandBuffer`. Caller is expected to commit +
    /// wait once all dispatches are encoded.
    pub fn new_command_buffer(&self) -> Option<CommandBuffer> {
        let cb = self.queue.commandBuffer()?;
        Some(CommandBuffer { inner: cb })
    }
}

fn load_library_from_blob(
    mtl: &ProtocolObject<dyn MTLDevice>,
    blob: &[u8],
    flavor: &str,
) -> Option<Retained<ProtocolObject<dyn MTLLibrary>>> {
    // `newLibraryWithData:error:` takes a `dispatch_data_t`. Wrapping
    // a Rust slice in `dispatch_data_create` would drag in a whole
    // `dispatch2` dep just for one call. Cheaper path: write the blob
    // to a temp file and use `newLibraryWithURL:error:`. The cost is
    // a few ms at startup — irrelevant next to model load.
    let seq = METALLIB_LOAD_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp = std::env::temp_dir().join(format!(
        "greppy_embed_native_{}_{}_{}.metallib",
        flavor,
        std::process::id(),
        seq,
    ));
    if let Err(e) = std::fs::write(&tmp, blob) {
        set_last_error(format!(
            "failed to write temp metallib to {}: {e}",
            tmp.display()
        ));
        return None;
    }
    let url = unsafe {
        objc2_foundation::NSURL::fileURLWithPath(&NSString::from_str(
            tmp.to_string_lossy().as_ref(),
        ))
    };
    let lib = unsafe { mtl.newLibraryWithURL_error(&url) }
        .map_err(|e| set_last_error(format!("newLibraryWithURL_error: {e:?}")))
        .ok()?;
    // Best-effort cleanup; not fatal if it fails.
    let _ = std::fs::remove_file(&tmp);
    Some(lib)
}

#[cfg(embed_native_has_tensor_metallib)]
fn maybe_load_tensor_library(
    mtl: &ProtocolObject<dyn MTLDevice>,
    supports_metal4: bool,
) -> Option<Retained<ProtocolObject<dyn MTLLibrary>>> {
    let force_simdgroup = env_truthy("EMBED_NATIVE_METAL_FORCE_SIMDGROUP")
        || env_truthy("EMBED_NATIVE_METAL_FORCE_FALLBACK");
    if !(supports_metal4 && !force_simdgroup) {
        return None;
    }

    if METALLIB_TENSOR_BLOB.is_empty() {
        set_last_error(
            "greppy-embed-native: tensor metallib blob empty; using Metal 3 simdgroup mul_mm",
        );
        return None;
    }

    match load_library_from_blob(mtl, METALLIB_TENSOR_BLOB, "tensor") {
        Some(lib) => Some(lib),
        None => {
            eprintln!(
                "greppy_embed_native: Metal tensor library unavailable, using simdgroup mul_mm: {}",
                crate::metal::errors::last_error()
            );
            None
        }
    }
}

#[cfg(not(embed_native_has_tensor_metallib))]
fn maybe_load_tensor_library(
    _mtl: &ProtocolObject<dyn MTLDevice>,
    _supports_metal4: bool,
) -> Option<Retained<ProtocolObject<dyn MTLLibrary>>> {
    None
}

fn is_tensor_mul_mm_kernel(kernel_name: &str) -> bool {
    kernel_name.starts_with("kernel_mul_mm_") && !kernel_name.starts_with("kernel_mul_mm_id")
}

fn env_truthy(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let v = v.trim();
            !v.is_empty()
                && !v.eq_ignore_ascii_case("0")
                && !v.eq_ignore_ascii_case("false")
                && !v.eq_ignore_ascii_case("no")
        }
        Err(_) => false,
    }
}

// ─── Buffer ────────────────────────────────────────────────────────

/// A shared-storage `MTLBuffer`. CPU-visible pointer lives at
/// `.as_ptr()`; the GPU reads from the same backing memory.
#[derive(Clone)]
pub struct Buffer {
    inner: Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
}

impl Buffer {
    pub fn len(&self) -> usize {
        self.inner.length() as usize
    }

    pub fn as_ptr(&self) -> *mut c_void {
        self.inner.contents().as_ptr()
    }

    /// Write a slice of `T` into the buffer starting at `byte_offset`.
    /// No bounds checking beyond the buffer length — the caller is
    /// expected to size the buffer correctly. `T` must be plain-old-data.
    pub unsafe fn write<T: Copy>(&self, byte_offset: usize, src: &[T]) {
        let dst = (self.as_ptr() as *mut u8).add(byte_offset);
        std::ptr::copy_nonoverlapping(src.as_ptr() as *const u8, dst, std::mem::size_of_val(src));
    }

    /// Read a slice of `T` out of the buffer starting at `byte_offset`.
    pub unsafe fn read<T: Copy>(&self, byte_offset: usize, dst: &mut [T]) {
        let src = (self.as_ptr() as *const u8).add(byte_offset);
        std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr() as *mut u8, std::mem::size_of_val(dst));
    }

    pub(crate) fn raw(&self) -> &ProtocolObject<dyn objc2_metal::MTLBuffer> {
        &self.inner
    }
}

// ─── CommandBuffer + ComputeEncoder ─────────────────────────────────

pub struct CommandBuffer {
    inner: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

pub struct SubmittedCommandBuffer {
    inner: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

impl CommandBuffer {
    /// Open a compute pass. Close it with `encoder.end()` before
    /// committing the command buffer.
    pub fn compute(&self) -> Option<ComputeEncoder> {
        let enc = self.inner.computeCommandEncoder()?;
        Some(ComputeEncoder { inner: enc })
    }

    /// Open a concurrent compute pass. Dispatches separated by
    /// `memoryBarrierWithScope:MTLBarrierScopeBuffers` preserve the graph
    /// dependencies; independent dispatches between barriers may overlap.
    pub fn compute_concurrent(&self) -> Option<ComputeEncoder> {
        let enc = self
            .inner
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)?;
        Some(ComputeEncoder { inner: enc })
    }

    pub fn blit(&self) -> Option<BlitEncoder> {
        let enc = self.inner.blitCommandEncoder()?;
        Some(BlitEncoder { inner: enc })
    }

    /// Commit without waiting. The returned handle can be waited on
    /// after dependent command buffers have also been queued.
    pub fn commit(self) -> SubmittedCommandBuffer {
        self.inner.commit();
        SubmittedCommandBuffer { inner: self.inner }
    }

    /// Commit the command buffer and block until the GPU is done.
    /// Returns the error slot (empty string on success).
    pub fn commit_and_wait(self) -> Result<(), String> {
        self.commit_and_wait_inner()
    }

    pub fn commit_and_wait_timed(self) -> Result<CommandBufferTiming, String> {
        let t0 = Instant::now();
        self.commit_and_wait_inner()?;
        let submit_wait_secs = t0.elapsed().as_secs_f64();
        let kernel_start = unsafe { self.inner.kernelStartTime() };
        let kernel_end = unsafe { self.inner.kernelEndTime() };
        let gpu_start = unsafe { self.inner.GPUStartTime() };
        let gpu_end = unsafe { self.inner.GPUEndTime() };
        Ok(CommandBufferTiming {
            submit_wait_secs,
            kernel_secs: positive_delta(kernel_start, kernel_end),
            gpu_secs: positive_delta(gpu_start, gpu_end),
        })
    }

    fn commit_and_wait_inner(&self) -> Result<(), String> {
        if use_wait_until_completed() {
            self.inner.commit();
            unsafe { self.inner.waitUntilCompleted() };
        } else {
            let (tx, rx) = mpsc::sync_channel(1);
            let block =
                block2::RcBlock::<dyn Fn(NonNull<ProtocolObject<dyn MTLCommandBuffer>>)>::new(
                    move |_| {
                        let _ = tx.send(());
                    },
                );
            unsafe {
                self.inner.addCompletedHandler(
                    (&*block
                        as *const block2::Block<
                            dyn Fn(NonNull<ProtocolObject<dyn MTLCommandBuffer>>),
                        >)
                        .cast_mut(),
                );
            }
            self.inner.commit();
            rx.recv()
                .map_err(|e| format!("Metal completion handler wait failed: {e}"))?;
        }
        if let Some(err) = unsafe { self.inner.error() } {
            let desc = err.localizedDescription();
            let s: &NSString = desc.as_ref();
            return Err(s.to_string());
        }
        Ok(())
    }
}

impl SubmittedCommandBuffer {
    pub fn wait(self) -> Result<(), String> {
        unsafe { self.inner.waitUntilCompleted() };
        if let Some(err) = unsafe { self.inner.error() } {
            let desc = err.localizedDescription();
            let s: &NSString = desc.as_ref();
            return Err(s.to_string());
        }
        Ok(())
    }
}

pub struct ComputeEncoder {
    inner: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
}

pub struct BlitEncoder {
    inner: Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>,
}

impl BlitEncoder {
    pub fn copy_buffer(
        &self,
        src: &Buffer,
        src_offset: usize,
        dst: &Buffer,
        dst_offset: usize,
        bytes: usize,
    ) {
        unsafe {
            self.inner
                .copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    src.raw(),
                    src_offset as NSUInteger,
                    dst.raw(),
                    dst_offset as NSUInteger,
                    bytes as NSUInteger,
                );
        }
    }

    pub fn end(self) {
        self.inner.endEncoding();
    }
}

impl ComputeEncoder {
    pub fn set_pipeline(&self, pso: &ProtocolObject<dyn MTLComputePipelineState>) {
        self.inner.setComputePipelineState(pso);
    }

    pub fn set_buffer(&self, index: usize, buf: &Buffer, offset: usize) {
        unsafe {
            self.inner.setBuffer_offset_atIndex(
                Some(buf.raw()),
                offset as NSUInteger,
                index as NSUInteger,
            );
        }
    }

    /// Inline a small constant (e.g. an `int`, `float`, or POD
    /// `#[repr(C)]` struct like `KargsCpy`) directly into the
    /// argument slot. Matches the `constant int& foo [[buffer(N)]]` /
    /// `constant ggml_metal_kargs_cpy & args [[buffer(0)]]` pattern
    /// on the shader side.
    ///
    /// SAFETY: `T` must be a POD-layout type (`#[repr(C)]` with no
    /// heap pointers and no `Drop` impl that matters at byte-copy
    /// time). Rust doesn't have a Pod trait in std so this is a
    /// convention; `bytemuck::Pod` would formalize it but is a
    /// dependency we don't want here. The caller is responsible.
    pub fn set_bytes<T>(&self, index: usize, value: &T) {
        let len = std::mem::size_of::<T>();
        unsafe {
            self.inner.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(value as *const T as *mut c_void),
                len as NSUInteger,
                index as NSUInteger,
            );
        }
    }

    /// Inline a dynamic-length POD array. Mirrors MLX's
    /// `CommandEncoder::set_vector_bytes(vec, index)`.
    pub fn set_bytes_slice<T: Copy>(&self, index: usize, slice: &[T]) {
        let len = std::mem::size_of_val(slice);
        let ptr = if slice.is_empty() {
            std::ptr::NonNull::dangling().as_ptr() as *mut c_void
        } else {
            slice.as_ptr() as *mut c_void
        };
        unsafe {
            self.inner.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(ptr),
                len as NSUInteger,
                index as NSUInteger,
            );
        }
    }

    /// Set threadgroup memory size (in bytes) for a given
    /// threadgroup-allocated buffer index. Mirrors
    /// `ggml_metal_encoder_set_threadgroup_memory_size`.
    pub fn set_threadgroup_memory_size(&self, bytes: usize, index: usize) {
        unsafe {
            self.inner
                .setThreadgroupMemoryLength_atIndex(bytes as NSUInteger, index as NSUInteger);
        }
    }

    /// `dispatchThreadgroups:threadsPerThreadgroup:` — takes grid
    /// measured in THREADGROUPS, not threads. Mirrors
    /// `ggml_metal_encoder_dispatch_threadgroups`.
    pub fn dispatch_threadgroups(&self, grid_tg: (usize, usize, usize), tg: (usize, usize, usize)) {
        DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
        let grid_mtl = MTLSize {
            width: grid_tg.0 as NSUInteger,
            height: grid_tg.1 as NSUInteger,
            depth: grid_tg.2 as NSUInteger,
        };
        let tg_mtl = MTLSize {
            width: tg.0 as NSUInteger,
            height: tg.1 as NSUInteger,
            depth: tg.2 as NSUInteger,
        };
        self.inner
            .dispatchThreadgroups_threadsPerThreadgroup(grid_mtl, tg_mtl);
    }

    pub fn dispatch(&self, grid: (usize, usize, usize), tg: (usize, usize, usize)) {
        DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
        let grid_mtl = MTLSize {
            width: grid.0 as NSUInteger,
            height: grid.1 as NSUInteger,
            depth: grid.2 as NSUInteger,
        };
        let tg_mtl = MTLSize {
            width: tg.0 as NSUInteger,
            height: tg.1 as NSUInteger,
            depth: tg.2 as NSUInteger,
        };
        self.inner
            .dispatchThreads_threadsPerThreadgroup(grid_mtl, tg_mtl);
    }

    pub fn memory_barrier_buffers(&self) {
        unsafe {
            self.inner.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        }
    }

    pub fn end(self) {
        self.inner.endEncoding();
    }
}

fn positive_delta(start: f64, end: f64) -> f64 {
    if start > 0.0 && end >= start {
        end - start
    } else {
        0.0
    }
}

fn use_wait_until_completed() -> bool {
    std::env::var_os("EMBED_NATIVE_METAL_WAIT_UNTIL_COMPLETED").is_some()
}

// ─── Global device accessor ─────────────────────────────────────────

static GLOBAL_DEVICE: OnceLock<Option<Device>> = OnceLock::new();

/// Shared device for the whole crate. First call initializes.
/// Returns `None` if the Metal stack is unavailable.
pub fn global_device() -> Option<&'static Device> {
    GLOBAL_DEVICE.get_or_init(Device::default_system).as_ref()
}

// ─── Tiny dead-code link to silence unused-warning ──────────────────

#[allow(dead_code)]
fn _type_size_check() {
    // If the crate compiles this module at all, the Metal bindings
    // are wired up correctly. Keep a dead reference to the FFI entry
    // points so rust-analyzer / cargo doc both see them as live.
    let _: Option<unsafe extern "C" fn() -> *mut c_void> = None;
    let _ = CStr::from_bytes_with_nul(b"\0").ok();
}
