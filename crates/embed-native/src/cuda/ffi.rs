use std::ffi::{c_char, c_void, CStr};
use std::ptr::NonNull;
use std::sync::OnceLock;

use libloading::Library;

use crate::{Error, Result};

const CUDA_BACKEND_UNAVAILABLE: i32 = 20000;

#[cfg(all(
    any(target_os = "linux", target_os = "windows"),
    embed_native_has_cuda_dylib
))]
const CUDA_DYLIB_BLOB: &[u8] = include_bytes!(env!("GREPPY_EMBED_NATIVE_CUDA_DYLIB"));

#[cfg(not(all(
    any(target_os = "linux", target_os = "windows"),
    embed_native_has_cuda_dylib
)))]
const CUDA_DYLIB_BLOB: &[u8] = &[];

type GpCudaErrorString = unsafe extern "C" fn(i32) -> *const c_char;
type GpCudaInit = unsafe extern "C" fn(i32, *mut *mut c_void, *mut *mut c_void) -> i32;
type GpCudaDestroy = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
type GpCudaMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> i32;
type GpCudaFree = unsafe extern "C" fn(*mut c_void) -> i32;
type GpCudaMemcpyH2DAsync =
    unsafe extern "C" fn(*mut c_void, *const c_void, usize, *mut c_void) -> i32;
type GpCudaMemcpyD2HAsync =
    unsafe extern "C" fn(*mut c_void, *const c_void, usize, *mut c_void) -> i32;
type GpCudaMemcpyD2DAsync =
    unsafe extern "C" fn(*mut c_void, *const c_void, usize, *mut c_void) -> i32;
type GpCudaMemsetAsync = unsafe extern "C" fn(*mut c_void, i32, usize, *mut c_void) -> i32;
type GpCudaStreamSync = unsafe extern "C" fn(*mut c_void) -> i32;
type GpCudaMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> i32;

type GpEmbedQ4k =
    unsafe extern "C" fn(*const c_void, *const u32, *mut f32, i32, i32, f32, *mut c_void) -> i32;
type GpEmbedQ6k =
    unsafe extern "C" fn(*const c_void, *const u32, *mut f32, i32, i32, f32, *mut c_void) -> i32;
type GpRmsNorm =
    unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32, i32, f32, *mut c_void) -> i32;
type GpRmsNormAdd = unsafe extern "C" fn(
    *const f32,
    *const f32,
    *const f32,
    *mut f32,
    i32,
    i32,
    f32,
    *mut c_void,
) -> i32;
type GpQwenRmsNorm =
    unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32, i32, f32, *mut c_void) -> i32;
type GpQwenRmsNormAdd = unsafe extern "C" fn(
    *const f32,
    *const f32,
    *const f32,
    *mut f32,
    i32,
    i32,
    f32,
    *mut c_void,
) -> i32;
type GpQwenAddRmsNorm = unsafe extern "C" fn(
    *const f32,
    *const f32,
    *const f32,
    *mut f32,
    *mut f32,
    i32,
    i32,
    f32,
    *mut c_void,
) -> i32;
type GpQwenCausalConv1dSilu =
    unsafe extern "C" fn(*mut f32, *const f32, *mut f32, i32, i32, *mut c_void) -> i32;
type GpQwenNormalizeLinearQk =
    unsafe extern "C" fn(*mut f32, *mut f32, i32, i32, f32, *mut c_void) -> i32;
type GpQwenDeinterleaveQGate = unsafe extern "C" fn(
    *const f32,
    *mut f32,
    *mut f32,
    i32,
    i32,
    i32,
    i32,
    i32,
    *mut c_void,
) -> i32;
type GpQwenSwiGlu = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32, *mut c_void) -> i32;
type GpQwenApplySiluGate = unsafe extern "C" fn(*mut f32, *const f32, i32, *mut c_void) -> i32;
type GpQwenApplySigmoidGate = unsafe extern "C" fn(*mut f32, *const f32, i32, *mut c_void) -> i32;
type GpQwenAdd = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32, *mut c_void) -> i32;
type GpQwenArgmax =
    unsafe extern "C" fn(*const f32, *mut u32, i32, *mut f32, *mut u32, i32, *mut c_void) -> i32;
type GpQwenDeltaNetDecode = unsafe extern "C" fn(
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *mut f32,
    *mut f32,
    i32,
    i32,
    *mut c_void,
) -> i32;
type GpQwenRopeDecode = unsafe extern "C" fn(*mut f32, i32, i32, i32, i32, f32, *mut c_void) -> i32;
type GpQwenCacheWrite =
    unsafe extern "C" fn(*const f32, *mut f32, i32, i32, i32, i32, *mut c_void) -> i32;
type GpQwenAttentionScoresDecode = unsafe extern "C" fn(
    *const f32,
    *const f32,
    *mut f32,
    i32,
    i32,
    i32,
    i32,
    i32,
    f32,
    *mut c_void,
) -> i32;
type GpQwenSoftmaxDecode = unsafe extern "C" fn(*mut f32, i32, i32, i32, *mut c_void) -> i32;
type GpQwenAttentionValuesDecode = unsafe extern "C" fn(
    *const f32,
    *const f32,
    *mut f32,
    i32,
    i32,
    i32,
    i32,
    i32,
    *mut c_void,
) -> i32;
type GpRmsNormHeads = unsafe extern "C" fn(
    *const f32,
    *const f32,
    *mut f32,
    i32,
    i32,
    i32,
    i32,
    i32,
    f32,
    *mut c_void,
) -> i32;
type GpSplitHeads =
    unsafe extern "C" fn(*const f32, *mut f32, i32, i32, i32, i32, i32, *mut c_void) -> i32;
type GpRopeNeox =
    unsafe extern "C" fn(*const f32, *mut f32, i32, i32, i32, i32, f32, *mut c_void) -> i32;
type GpAttentionScores =
    unsafe extern "C" fn(*mut c_void, *const f32, *const f32, *mut f32, i32, i32, i32, i32) -> i32;
type GpSoftmaxMask =
    unsafe extern "C" fn(*mut f32, *const u32, i32, i32, i32, i32, f32, *mut c_void) -> i32;
type GpAttentionValues =
    unsafe extern "C" fn(*mut c_void, *const f32, *const f32, *mut f32, i32, i32, i32, i32) -> i32;
type GpMergeHeads =
    unsafe extern "C" fn(*const f32, *mut f32, i32, i32, i32, i32, *mut c_void) -> i32;
type GpGeglu = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32, *mut c_void) -> i32;
type GpMeanPool =
    unsafe extern "C" fn(*const f32, *const u32, *mut f32, i32, i32, i32, *mut c_void) -> i32;
type GpL2Norm = unsafe extern "C" fn(*const f32, *mut f32, i32, i32, *mut c_void) -> i32;
type GpMmqMatmul = unsafe extern "C" fn(
    i32,
    *const c_void,
    *const f32,
    *mut f32,
    *mut c_void,
    *mut c_void,
    i64,
    i64,
    i64,
    i64,
    *mut c_void,
) -> i32;
type GpMmvqMatvec = unsafe extern "C" fn(
    i32,
    *const c_void,
    *const f32,
    *mut f32,
    *mut c_void,
    i64,
    i64,
    i64,
    *mut c_void,
) -> i32;

struct CudaApi {
    _lib: Library,
    gp_cuda_error_string: GpCudaErrorString,
    gp_cuda_init: GpCudaInit,
    gp_cuda_destroy: GpCudaDestroy,
    gp_cuda_malloc: GpCudaMalloc,
    gp_cuda_free: GpCudaFree,
    gp_cuda_memcpy_h2d_async: GpCudaMemcpyH2DAsync,
    gp_cuda_memcpy_d2h_async: GpCudaMemcpyD2HAsync,
    gp_cuda_memcpy_d2d_async: GpCudaMemcpyD2DAsync,
    gp_cuda_memset_async: GpCudaMemsetAsync,
    gp_cuda_stream_sync: GpCudaStreamSync,
    gp_cuda_mem_get_info: GpCudaMemGetInfo,
    gp_embed_q4k: GpEmbedQ4k,
    gp_embed_q6k: GpEmbedQ6k,
    gp_rms_norm: GpRmsNorm,
    gp_rms_norm_add: GpRmsNormAdd,
    gp_qwen_rms_norm: GpQwenRmsNorm,
    gp_qwen_rms_norm_add: GpQwenRmsNormAdd,
    gp_qwen_add_rms_norm: GpQwenAddRmsNorm,
    gp_qwen_causal_conv1d_silu: GpQwenCausalConv1dSilu,
    gp_qwen_normalize_linear_qk: GpQwenNormalizeLinearQk,
    gp_qwen_deinterleave_q_gate: GpQwenDeinterleaveQGate,
    gp_qwen_swiglu: GpQwenSwiGlu,
    gp_qwen_apply_silu_gate: GpQwenApplySiluGate,
    gp_qwen_apply_sigmoid_gate: GpQwenApplySigmoidGate,
    gp_qwen_add: GpQwenAdd,
    gp_qwen_argmax: GpQwenArgmax,
    gp_qwen_deltanet_decode: GpQwenDeltaNetDecode,
    gp_qwen_rope_decode: GpQwenRopeDecode,
    gp_qwen_cache_write: GpQwenCacheWrite,
    gp_qwen_attention_scores_decode: GpQwenAttentionScoresDecode,
    gp_qwen_softmax_decode: GpQwenSoftmaxDecode,
    gp_qwen_attention_values_decode: GpQwenAttentionValuesDecode,
    gp_rms_norm_heads: GpRmsNormHeads,
    gp_split_heads: GpSplitHeads,
    gp_rope_neox: GpRopeNeox,
    gp_attention_scores: GpAttentionScores,
    gp_softmax_mask: GpSoftmaxMask,
    gp_attention_values: GpAttentionValues,
    gp_merge_heads: GpMergeHeads,
    gp_geglu: GpGeglu,
    gp_mean_pool: GpMeanPool,
    gp_l2_norm: GpL2Norm,
    gp_mmq_matmul: GpMmqMatmul,
    gp_mmvq_matvec: GpMmvqMatvec,
}

unsafe impl Send for CudaApi {}
unsafe impl Sync for CudaApi {}

static CUDA_API: OnceLock<std::result::Result<CudaApi, String>> = OnceLock::new();

fn cuda_api() -> Result<&'static CudaApi> {
    match CUDA_API.get_or_init(load_cuda_api) {
        Ok(api) => Ok(api),
        Err(err) => Err(Error::InvalidGguf(format!(
            "CUDA backend unavailable: {err}"
        ))),
    }
}

fn load_cuda_api() -> std::result::Result<CudaApi, String> {
    let lib = load_cuda_library()?;
    unsafe {
        Ok(CudaApi {
            gp_cuda_error_string: load_symbol(&lib, b"gp_cuda_error_string\0")?,
            gp_cuda_init: load_symbol(&lib, b"gp_cuda_init\0")?,
            gp_cuda_destroy: load_symbol(&lib, b"gp_cuda_destroy\0")?,
            gp_cuda_malloc: load_symbol(&lib, b"gp_cuda_malloc\0")?,
            gp_cuda_free: load_symbol(&lib, b"gp_cuda_free\0")?,
            gp_cuda_memcpy_h2d_async: load_symbol(&lib, b"gp_cuda_memcpy_h2d_async\0")?,
            gp_cuda_memcpy_d2h_async: load_symbol(&lib, b"gp_cuda_memcpy_d2h_async\0")?,
            gp_cuda_memcpy_d2d_async: load_symbol(&lib, b"gp_cuda_memcpy_d2d_async\0")?,
            gp_cuda_memset_async: load_symbol(&lib, b"gp_cuda_memset_async\0")?,
            gp_cuda_stream_sync: load_symbol(&lib, b"gp_cuda_stream_sync\0")?,
            gp_cuda_mem_get_info: load_symbol(&lib, b"gp_cuda_mem_get_info\0")?,
            gp_embed_q4k: load_symbol(&lib, b"gp_embed_q4k\0")?,
            gp_embed_q6k: load_symbol(&lib, b"gp_embed_q6k\0")?,
            gp_rms_norm: load_symbol(&lib, b"gp_rms_norm\0")?,
            gp_rms_norm_add: load_symbol(&lib, b"gp_rms_norm_add\0")?,
            gp_qwen_rms_norm: load_symbol(&lib, b"gp_qwen_rms_norm\0")?,
            gp_qwen_rms_norm_add: load_symbol(&lib, b"gp_qwen_rms_norm_add\0")?,
            gp_qwen_add_rms_norm: load_symbol(&lib, b"gp_qwen_add_rms_norm\0")?,
            gp_qwen_causal_conv1d_silu: load_symbol(&lib, b"gp_qwen_causal_conv1d_silu\0")?,
            gp_qwen_normalize_linear_qk: load_symbol(&lib, b"gp_qwen_normalize_linear_qk\0")?,
            gp_qwen_deinterleave_q_gate: load_symbol(&lib, b"gp_qwen_deinterleave_q_gate\0")?,
            gp_qwen_swiglu: load_symbol(&lib, b"gp_qwen_swiglu\0")?,
            gp_qwen_apply_silu_gate: load_symbol(&lib, b"gp_qwen_apply_silu_gate\0")?,
            gp_qwen_apply_sigmoid_gate: load_symbol(&lib, b"gp_qwen_apply_sigmoid_gate\0")?,
            gp_qwen_add: load_symbol(&lib, b"gp_qwen_add\0")?,
            gp_qwen_argmax: load_symbol(&lib, b"gp_qwen_argmax\0")?,
            gp_qwen_deltanet_decode: load_symbol(&lib, b"gp_qwen_deltanet_decode\0")?,
            gp_qwen_rope_decode: load_symbol(&lib, b"gp_qwen_rope_decode\0")?,
            gp_qwen_cache_write: load_symbol(&lib, b"gp_qwen_cache_write\0")?,
            gp_qwen_attention_scores_decode: load_symbol(
                &lib,
                b"gp_qwen_attention_scores_decode\0",
            )?,
            gp_qwen_softmax_decode: load_symbol(&lib, b"gp_qwen_softmax_decode\0")?,
            gp_qwen_attention_values_decode: load_symbol(
                &lib,
                b"gp_qwen_attention_values_decode\0",
            )?,
            gp_rms_norm_heads: load_symbol(&lib, b"gp_rms_norm_heads\0")?,
            gp_split_heads: load_symbol(&lib, b"gp_split_heads\0")?,
            gp_rope_neox: load_symbol(&lib, b"gp_rope_neox\0")?,
            gp_attention_scores: load_symbol(&lib, b"gp_attention_scores\0")?,
            gp_softmax_mask: load_symbol(&lib, b"gp_softmax_mask\0")?,
            gp_attention_values: load_symbol(&lib, b"gp_attention_values\0")?,
            gp_merge_heads: load_symbol(&lib, b"gp_merge_heads\0")?,
            gp_geglu: load_symbol(&lib, b"gp_geglu\0")?,
            gp_mean_pool: load_symbol(&lib, b"gp_mean_pool\0")?,
            gp_l2_norm: load_symbol(&lib, b"gp_l2_norm\0")?,
            gp_mmq_matmul: load_symbol(&lib, b"gp_mmq_matmul\0")?,
            gp_mmvq_matvec: load_symbol(&lib, b"gp_mmvq_matvec\0")?,
            _lib: lib,
        })
    }
}

fn load_cuda_library() -> std::result::Result<Library, String> {
    if let Ok(path) = std::env::var("EMBED_NATIVE_CUDA_LIBRARY") {
        let path = path.trim();
        if !path.is_empty() {
            return unsafe { Library::new(path) }
                .map_err(|e| format!("failed to load {path}: {e}"));
        }
    }

    if CUDA_DYLIB_BLOB.is_empty() {
        return Err("CUDA backend library was not built into this binary".into());
    }

    let ext = if cfg!(target_os = "windows") {
        "dll"
    } else {
        "so"
    };
    let path = std::env::temp_dir().join(format!(
        "greppy_embed_native_cuda_{}.{}",
        std::process::id(),
        ext
    ));
    std::fs::write(&path, CUDA_DYLIB_BLOB).map_err(|e| {
        format!(
            "failed to write bundled CUDA backend {}: {e}",
            path.display()
        )
    })?;
    unsafe { Library::new(&path) }.map_err(|e| {
        format!(
            "failed to load bundled CUDA backend {}: {e}",
            path.display()
        )
    })
}

unsafe fn load_symbol<T: Copy>(lib: &Library, name: &[u8]) -> std::result::Result<T, String> {
    let symbol = unsafe { lib.get::<T>(name) }.map_err(|e| {
        format!(
            "missing CUDA backend symbol {}: {e}",
            String::from_utf8_lossy(name).trim_end_matches('\0')
        )
    })?;
    Ok(*symbol)
}

pub struct CudaDevice {
    stream: NonNull<c_void>,
    blas: NonNull<c_void>,
}

unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

impl CudaDevice {
    pub fn new(device: i32) -> Result<Self> {
        let api = cuda_api()?;
        let mut stream = std::ptr::null_mut();
        let mut blas = std::ptr::null_mut();
        check(
            unsafe { (api.gp_cuda_init)(device, &mut stream, &mut blas) },
            "cuda init",
        )?;
        let stream = NonNull::new(stream)
            .ok_or_else(|| Error::InvalidGguf("CUDA stream creation returned null".into()))?;
        let blas = NonNull::new(blas)
            .ok_or_else(|| Error::InvalidGguf("cuBLAS creation returned null".into()))?;
        Ok(Self { stream, blas })
    }

    pub fn stream(&self) -> *mut c_void {
        self.stream.as_ptr()
    }

    pub fn blas(&self) -> *mut c_void {
        self.blas.as_ptr()
    }

    pub fn sync(&self) -> Result<()> {
        let api = cuda_api()?;
        check(
            unsafe { (api.gp_cuda_stream_sync)(self.stream()) },
            "cuda stream sync",
        )
    }

    pub fn mem_info(&self) -> Result<(usize, usize)> {
        let api = cuda_api()?;
        let mut free = 0usize;
        let mut total = 0usize;
        check(
            unsafe { (api.gp_cuda_mem_get_info)(&mut free, &mut total) },
            "cuda mem info",
        )?;
        Ok((free, total))
    }

    pub fn alloc(&self, bytes: usize) -> Result<DeviceBuffer> {
        let api = cuda_api()?;
        let mut ptr = std::ptr::null_mut();
        check(
            unsafe { (api.gp_cuda_malloc)(&mut ptr, bytes) },
            "cuda malloc",
        )?;
        let ptr = NonNull::new(ptr).ok_or_else(|| {
            Error::InvalidGguf(format!("cuda malloc returned null ({bytes} bytes)"))
        })?;
        Ok(DeviceBuffer { ptr, bytes })
    }

    pub fn upload_bytes(&self, bytes: &[u8]) -> Result<DeviceBuffer> {
        let buf = self.alloc(bytes.len().max(1))?;
        self.copy_h2d(&buf, bytes)?;
        Ok(buf)
    }

    pub fn copy_h2d<T>(&self, dst: &DeviceBuffer, src: &[T]) -> Result<()> {
        let api = cuda_api()?;
        let bytes = std::mem::size_of_val(src);
        if bytes > dst.bytes {
            return Err(Error::InvalidGguf(format!(
                "cuda h2d copy {bytes} bytes exceeds dst {} bytes",
                dst.bytes
            )));
        }
        check(
            unsafe {
                (api.gp_cuda_memcpy_h2d_async)(
                    dst.ptr(),
                    src.as_ptr() as *const c_void,
                    bytes,
                    self.stream(),
                )
            },
            "cuda memcpy h2d",
        )
    }

    pub fn copy_d2h<T>(&self, dst: &mut [T], src: &DeviceBuffer) -> Result<()> {
        let api = cuda_api()?;
        let bytes = std::mem::size_of_val(dst);
        if bytes > src.bytes {
            return Err(Error::InvalidGguf(format!(
                "cuda d2h copy {bytes} bytes exceeds src {} bytes",
                src.bytes
            )));
        }
        check(
            unsafe {
                (api.gp_cuda_memcpy_d2h_async)(
                    dst.as_mut_ptr() as *mut c_void,
                    src.ptr(),
                    bytes,
                    self.stream(),
                )
            },
            "cuda memcpy d2h",
        )?;
        self.sync()
    }

    pub fn copy_d2d(&self, dst: &DeviceBuffer, src: &DeviceBuffer, bytes: usize) -> Result<()> {
        let api = cuda_api()?;
        if bytes > dst.bytes || bytes > src.bytes {
            return Err(Error::InvalidGguf(format!(
                "cuda d2d copy {bytes} bytes exceeds dst {} or src {} bytes",
                dst.bytes, src.bytes
            )));
        }
        check(
            unsafe { (api.gp_cuda_memcpy_d2d_async)(dst.ptr(), src.ptr(), bytes, self.stream()) },
            "cuda memcpy d2d",
        )
    }

    pub fn memset(&self, dst: &DeviceBuffer, value: i32) -> Result<()> {
        let api = cuda_api()?;
        check(
            unsafe { (api.gp_cuda_memset_async)(dst.ptr(), value, dst.bytes, self.stream()) },
            "cuda memset",
        )
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        if let Ok(api) = cuda_api() {
            let _ = unsafe { (api.gp_cuda_destroy)(self.stream(), self.blas()) };
        }
    }
}

pub struct DeviceBuffer {
    ptr: NonNull<c_void>,
    bytes: usize,
}

unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    pub fn ptr(&self) -> *mut c_void {
        self.ptr.as_ptr()
    }

    pub fn as_f32(&self) -> *mut f32 {
        self.ptr() as *mut f32
    }

    pub fn as_u32(&self) -> *mut u32 {
        self.ptr() as *mut u32
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if let Ok(api) = cuda_api() {
            let _ = unsafe { (api.gp_cuda_free)(self.ptr()) };
        }
    }
}

pub fn check(code: i32, what: &str) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    let msg = cuda_error_string(code);
    Err(Error::InvalidGguf(format!("{what}: {msg} ({code})")))
}

fn cuda_error_string(code: i32) -> String {
    if code == CUDA_BACKEND_UNAVAILABLE {
        return "CUDA backend unavailable".into();
    }
    let Ok(api) = cuda_api() else {
        return format!("CUDA error code {code}");
    };
    unsafe {
        let raw = (api.gp_cuda_error_string)(code);
        if raw.is_null() {
            format!("CUDA error code {code}")
        } else {
            CStr::from_ptr(raw).to_string_lossy().into_owned()
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_embed_q4k(
    weights: *const c_void,
    ids: *const u32,
    dst: *mut f32,
    rows: i32,
    hidden: i32,
    scale: f32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe { (api.gp_embed_q4k)(weights, ids, dst, rows, hidden, scale, stream) },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_embed_q6k(
    weights: *const c_void,
    ids: *const u32,
    dst: *mut f32,
    rows: i32,
    hidden: i32,
    scale: f32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe { (api.gp_embed_q6k)(weights, ids, dst, rows, hidden, scale, stream) },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_q4k_dequantizes_one_row_on_cuda() {
        let dev = CudaDevice::new(0).expect("CUDA device");
        let mut raw = Vec::with_capacity(144);
        raw.extend_from_slice(&0x3c00u16.to_le_bytes());
        raw.extend_from_slice(&0u16.to_le_bytes());
        raw.extend_from_slice(&[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]);
        for i in 0..128u8 {
            let lo = i & 0x0f;
            let hi = 15 - lo;
            raw.push(lo | (hi << 4));
        }
        let weights = dev.upload_bytes(&raw).expect("upload q4k");
        let ids = dev.upload_bytes(&0u32.to_le_bytes()).expect("upload ids");
        let dst = dev.alloc(256 * std::mem::size_of::<f32>()).unwrap();

        check(
            unsafe {
                gp_embed_q4k(
                    weights.ptr(),
                    ids.as_u32(),
                    dst.as_f32(),
                    1,
                    256,
                    1.0,
                    dev.stream(),
                )
            },
            "cuda embed_q4k",
        )
        .unwrap();

        let mut out = vec![0.0f32; 256];
        dev.copy_d2h(&mut out, &dst).unwrap();
        for sub in 0..8usize {
            for lane in 0..32usize {
                let idx = sub * 32 + lane;
                let packed = raw[16 + (sub / 2) * 32 + lane];
                let expected = if sub % 2 == 0 {
                    (packed & 0x0f) as f32
                } else {
                    (packed >> 4) as f32
                };
                assert_eq!(out[idx], expected, "idx={idx}");
            }
        }
    }

    #[test]
    fn mmvq_q8_0_matvec_matches_known_dot_on_cuda() {
        let dev = CudaDevice::new(0).expect("CUDA device");
        let mut raw = Vec::with_capacity(2 * 34);
        for row in 0..2i8 {
            raw.extend_from_slice(&0x3c00u16.to_le_bytes());
            for i in 0..32i8 {
                raw.push((i + row) as u8);
            }
        }
        let weights = dev.upload_bytes(&raw).expect("upload q8_0");
        let input = (0..32).map(|i| i as f32 / 31.0).collect::<Vec<_>>();
        let src = dev.alloc(std::mem::size_of_val(&input[..])).unwrap();
        dev.copy_h2d(&src, &input).unwrap();
        let dst = dev.alloc(2 * std::mem::size_of::<f32>()).unwrap();
        let scratch = dev.alloc(36 + 128 * 144).unwrap();
        dev.memset(&scratch, 0).unwrap();
        check(
            unsafe {
                gp_mmvq_matvec(
                    8,
                    weights.ptr(),
                    src.as_f32(),
                    dst.as_f32(),
                    scratch.ptr(),
                    32,
                    1,
                    2,
                    dev.stream(),
                )
            },
            "cuda mmvq q8_0",
        )
        .unwrap();
        let mut out = [0.0f32; 2];
        dev.copy_d2h(&mut out, &dst).unwrap();
        for row in 0..2 {
            let expected = input
                .iter()
                .enumerate()
                .map(|(i, v)| ((i + row) as f32) * *v)
                .sum::<f32>();
            assert!(
                (out[row] - expected).abs() < 1.0,
                "row={row} out={} expected={expected}",
                out[row]
            );
        }
    }

    #[test]
    fn mmvq_q4k_matvec_matches_known_dot_on_cuda() {
        let dev = CudaDevice::new(0).expect("CUDA device");
        let mut raw = Vec::with_capacity(2 * 144);
        let mut expected = [0.0f32; 2];
        for row in 0..2u8 {
            raw.extend_from_slice(&0x3c00u16.to_le_bytes());
            raw.extend_from_slice(&0u16.to_le_bytes());
            raw.extend_from_slice(&[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]);
            for i in 0..128u8 {
                let lo = (i + row) & 0x0f;
                let hi = 15u8.wrapping_sub(i.wrapping_add(row)) & 0x0f;
                expected[row as usize] += lo as f32 + hi as f32;
                raw.push(lo | (hi << 4));
            }
        }
        let weights = dev.upload_bytes(&raw).expect("upload q4k");
        let input = vec![1.0f32; 256];
        let src = dev.alloc(std::mem::size_of_val(&input[..])).unwrap();
        dev.copy_h2d(&src, &input).unwrap();
        let dst = dev.alloc(2 * std::mem::size_of::<f32>()).unwrap();
        let scratch = dev.alloc(16 * 36 + 128 * 144).unwrap();
        dev.memset(&scratch, 0).unwrap();
        check(
            unsafe {
                gp_mmvq_matvec(
                    12,
                    weights.ptr(),
                    src.as_f32(),
                    dst.as_f32(),
                    scratch.ptr(),
                    256,
                    1,
                    2,
                    dev.stream(),
                )
            },
            "cuda mmvq q4k",
        )
        .unwrap();
        let mut out = [0.0f32; 2];
        dev.copy_d2h(&mut out, &dst).unwrap();
        for row in 0..2 {
            assert!(
                (out[row] - expected[row]).abs() < 2.0,
                "row={row} out={} expected={}",
                out[row],
                expected[row]
            );
        }
    }

    #[test]
    fn mmvq_q5k_matvec_matches_known_dot_on_cuda() {
        let dev = CudaDevice::new(0).expect("CUDA device");
        let blocks_per_row = 4usize;
        let mut raw = Vec::with_capacity(2 * blocks_per_row * 176);
        let mut expected = [0.0f32; 2];
        for row in 0..2u8 {
            for block in 0..blocks_per_row as u8 {
                raw.extend_from_slice(&0x3c00u16.to_le_bytes());
                raw.extend_from_slice(&0u16.to_le_bytes());
                raw.extend_from_slice(&[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]);
                raw.extend_from_slice(&[0xffu8; 32]);
                for i in 0..128u8 {
                    let lo = (i + row + block) & 0x0f;
                    let hi = 15u8.wrapping_sub(i.wrapping_add(row).wrapping_add(block)) & 0x0f;
                    expected[row as usize] += (lo | 0x10) as f32 + (hi | 0x10) as f32;
                    raw.push(lo | (hi << 4));
                }
            }
        }
        let weights = dev.upload_bytes(&raw).expect("upload q5k");
        let input = vec![1.0f32; blocks_per_row * 256];
        let src = dev.alloc(std::mem::size_of_val(&input[..])).unwrap();
        dev.copy_h2d(&src, &input).unwrap();
        let dst = dev.alloc(2 * std::mem::size_of::<f32>()).unwrap();
        let scratch = dev.alloc(32 * 36 + 128 * 144).unwrap();
        dev.memset(&scratch, 0).unwrap();
        check(
            unsafe {
                gp_mmvq_matvec(
                    13,
                    weights.ptr(),
                    src.as_f32(),
                    dst.as_f32(),
                    scratch.ptr(),
                    (blocks_per_row * 256) as i64,
                    blocks_per_row as i64,
                    2,
                    dev.stream(),
                )
            },
            "cuda mmvq q5k",
        )
        .unwrap();
        let mut out = [0.0f32; 2];
        dev.copy_d2h(&mut out, &dst).unwrap();
        for row in 0..2 {
            assert!(
                (out[row] - expected[row]).abs() < 2.0,
                "row={row} out={} expected={}",
                out[row],
                expected[row]
            );
        }
    }

    #[test]
    fn mmq_q5k_matmul_matches_known_dot_on_cuda() {
        let dev = CudaDevice::new(0).expect("CUDA device");
        let blocks_per_row = 4usize;
        let mut raw = Vec::with_capacity(2 * blocks_per_row * 176);
        let mut expected = [0.0f32; 2];
        for row in 0..2u8 {
            for block in 0..blocks_per_row as u8 {
                raw.extend_from_slice(&0x3c00u16.to_le_bytes());
                raw.extend_from_slice(&0u16.to_le_bytes());
                raw.extend_from_slice(&[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]);
                raw.extend_from_slice(&[0xffu8; 32]);
                for i in 0..128u8 {
                    let lo = (i + row + block) & 0x0f;
                    let hi = 15u8.wrapping_sub(i.wrapping_add(row).wrapping_add(block)) & 0x0f;
                    expected[row as usize] += (lo | 0x10) as f32 + (hi | 0x10) as f32;
                    raw.push(lo | (hi << 4));
                }
            }
        }
        let weights = dev.upload_bytes(&raw).expect("upload q5k");
        let input = vec![1.0f32; blocks_per_row * 256];
        let src = dev.alloc(std::mem::size_of_val(&input[..])).unwrap();
        dev.copy_h2d(&src, &input).unwrap();
        let dst = dev.alloc(2 * std::mem::size_of::<f32>()).unwrap();
        let q8_scratch = dev.alloc(32 * 36 + 128 * 144).unwrap();
        dev.memset(&q8_scratch, 0).unwrap();
        let fixup_scratch = dev
            .alloc(128 * 128 * 128 * std::mem::size_of::<f32>())
            .unwrap();
        dev.memset(&fixup_scratch, 0).unwrap();
        check(
            unsafe {
                gp_mmq_matmul(
                    13,
                    weights.ptr(),
                    src.as_f32(),
                    dst.as_f32(),
                    q8_scratch.ptr(),
                    fixup_scratch.ptr(),
                    (blocks_per_row * 256) as i64,
                    blocks_per_row as i64,
                    2,
                    1,
                    dev.stream(),
                )
            },
            "cuda mmq q5k",
        )
        .unwrap();
        let mut out = [0.0f32; 2];
        dev.copy_d2h(&mut out, &dst).unwrap();
        for row in 0..2 {
            assert!(
                (out[row] - expected[row]).abs() < 2.0,
                "row={row} out={} expected={}",
                out[row],
                expected[row]
            );
        }
    }

    #[test]
    fn qwen_rms_norm_uses_gguf_weight_on_cuda() {
        let dev = CudaDevice::new(0).expect("CUDA device");
        let src_host = [3.0f32, 4.0, -2.0, 1.0, -1.5, 0.5, 2.5, -3.5];
        let weight_host = [0.25f32, -0.5, 1.0, 0.0];
        let src = dev.alloc(std::mem::size_of_val(&src_host)).unwrap();
        let weight = dev.alloc(std::mem::size_of_val(&weight_host)).unwrap();
        let dst = dev.alloc(std::mem::size_of_val(&src_host)).unwrap();
        dev.copy_h2d(&src, &src_host).unwrap();
        dev.copy_h2d(&weight, &weight_host).unwrap();
        check(
            unsafe {
                gp_qwen_rms_norm(
                    src.as_f32(),
                    weight.as_f32(),
                    dst.as_f32(),
                    2,
                    4,
                    1.0e-6,
                    dev.stream(),
                )
            },
            "cuda qwen_rms_norm",
        )
        .unwrap();

        let mut out = [0.0f32; 8];
        dev.copy_d2h(&mut out, &dst).unwrap();
        for row in 0..2 {
            let values = &src_host[row * 4..row * 4 + 4];
            let mean_sq = values.iter().map(|v| v * v).sum::<f32>() / 4.0;
            let inv = 1.0 / (mean_sq + 1.0e-6).sqrt();
            for col in 0..4 {
                let expected = values[col] * inv * weight_host[col];
                let actual = out[row * 4 + col];
                assert!(
                    (actual - expected).abs() <= 2.0e-6,
                    "row={row} col={col} actual={actual} expected={expected}"
                );
            }
        }
    }

    #[test]
    fn qwen_add_rms_norm_matches_cpu_on_cuda() {
        let dev = CudaDevice::new(0).expect("CUDA device");
        let lhs_host = [3.0f32, 4.0, -2.0, 1.0, -1.5, 0.5, 2.5, -3.5];
        let rhs_host = [-1.0f32, 0.25, 0.5, 2.0, 1.25, -0.75, 0.5, 1.0];
        let weight_host = [0.25f32, -0.5, 1.0, 0.0];
        let lhs = dev.alloc(std::mem::size_of_val(&lhs_host)).unwrap();
        let rhs = dev.alloc(std::mem::size_of_val(&rhs_host)).unwrap();
        let weight = dev.alloc(std::mem::size_of_val(&weight_host)).unwrap();
        let sum = dev.alloc(std::mem::size_of_val(&lhs_host)).unwrap();
        let norm = dev.alloc(std::mem::size_of_val(&lhs_host)).unwrap();
        dev.copy_h2d(&lhs, &lhs_host).unwrap();
        dev.copy_h2d(&rhs, &rhs_host).unwrap();
        dev.copy_h2d(&weight, &weight_host).unwrap();
        check(
            unsafe {
                gp_qwen_add_rms_norm(
                    lhs.as_f32(),
                    rhs.as_f32(),
                    weight.as_f32(),
                    sum.as_f32(),
                    norm.as_f32(),
                    2,
                    4,
                    1.0e-6,
                    dev.stream(),
                )
            },
            "cuda qwen_add_rms_norm",
        )
        .unwrap();

        let mut sum_out = [0.0f32; 8];
        let mut norm_out = [0.0f32; 8];
        dev.copy_d2h(&mut sum_out, &sum).unwrap();
        dev.copy_d2h(&mut norm_out, &norm).unwrap();
        for row in 0..2 {
            let base = row * 4;
            let mut values = [0.0f32; 4];
            for col in 0..4 {
                values[col] = lhs_host[base + col] + rhs_host[base + col];
                assert!(
                    (sum_out[base + col] - values[col]).abs() <= 1.0e-6,
                    "sum row={row} col={col} actual={} expected={}",
                    sum_out[base + col],
                    values[col]
                );
            }
            let mean_sq = values.iter().map(|v| v * v).sum::<f32>() / 4.0;
            let inv = 1.0 / (mean_sq + 1.0e-6).sqrt();
            for col in 0..4 {
                let expected = values[col] * inv * weight_host[col];
                let actual = norm_out[base + col];
                assert!(
                    (actual - expected).abs() <= 2.0e-6,
                    "norm row={row} col={col} actual={actual} expected={expected}"
                );
            }
        }
    }

    #[test]
    fn qwen_causal_conv1d_silu_matches_cpu_on_cuda() {
        let dev = CudaDevice::new(0).expect("CUDA device");
        let mut values_host = [1.0f32, -2.0, 0.5];
        let weights_host = [
            0.25f32, -0.5, 0.75, 1.0, -1.0, 0.5, 0.25, -0.25, 0.1, 0.2, -0.3, 0.4,
        ];
        let mut state_host = [
            0.5f32, -0.5, 1.0, 0.0, -1.0, 2.0, -2.0, 1.0, 0.25, -0.25, 0.75, -0.75,
        ];
        let mut expected_values = values_host;
        let mut expected_state = state_host;
        cpu_causal_conv1d_silu(&mut expected_values, &weights_host, &mut expected_state, 4);

        let values = dev.alloc(std::mem::size_of_val(&values_host)).unwrap();
        let weights = dev.alloc(std::mem::size_of_val(&weights_host)).unwrap();
        let state = dev.alloc(std::mem::size_of_val(&state_host)).unwrap();
        dev.copy_h2d(&values, &values_host).unwrap();
        dev.copy_h2d(&weights, &weights_host).unwrap();
        dev.copy_h2d(&state, &state_host).unwrap();
        check(
            unsafe {
                gp_qwen_causal_conv1d_silu(
                    values.as_f32(),
                    weights.as_f32(),
                    state.as_f32(),
                    3,
                    4,
                    dev.stream(),
                )
            },
            "cuda qwen causal_conv1d_silu",
        )
        .unwrap();
        dev.copy_d2h(&mut values_host, &values).unwrap();
        dev.copy_d2h(&mut state_host, &state).unwrap();
        for (idx, (actual, expected)) in values_host.iter().zip(expected_values).enumerate() {
            assert!(
                (*actual - expected).abs() <= 2.0e-6,
                "value idx={idx} actual={actual} expected={expected}"
            );
        }
        for (idx, (actual, expected)) in state_host.iter().zip(expected_state).enumerate() {
            assert_eq!(*actual, expected, "state idx={idx}");
        }
    }

    #[test]
    fn qwen_normalize_linear_qk_matches_cpu_on_cuda() {
        let dev = CudaDevice::new(0).expect("CUDA device");
        let mut q_host = [1.0f32, -2.0, 3.0, -4.0, 0.5, 1.5, -2.5, 3.5];
        let mut k_host = [-1.0f32, 2.5, -3.5, 4.5, 2.0, -1.0, 0.25, -0.75];
        let mut expected_q = q_host;
        let mut expected_k = k_host;
        cpu_normalize_linear_qk(&mut expected_q, &mut expected_k, 2, 4, 1.0e-6);

        let q = dev.alloc(std::mem::size_of_val(&q_host)).unwrap();
        let k = dev.alloc(std::mem::size_of_val(&k_host)).unwrap();
        dev.copy_h2d(&q, &q_host).unwrap();
        dev.copy_h2d(&k, &k_host).unwrap();
        check(
            unsafe {
                gp_qwen_normalize_linear_qk(q.as_f32(), k.as_f32(), 2, 4, 1.0e-6, dev.stream())
            },
            "cuda qwen normalize_linear_qk",
        )
        .unwrap();
        dev.copy_d2h(&mut q_host, &q).unwrap();
        dev.copy_d2h(&mut k_host, &k).unwrap();
        for (idx, (actual, expected)) in q_host.iter().zip(expected_q).enumerate() {
            assert!(
                (*actual - expected).abs() <= 2.0e-6,
                "q idx={idx} actual={actual} expected={expected}"
            );
        }
        for (idx, (actual, expected)) in k_host.iter().zip(expected_k).enumerate() {
            assert!(
                (*actual - expected).abs() <= 2.0e-6,
                "k idx={idx} actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn qwen_swiglu_and_silu_gate_match_cpu_on_cuda() {
        let dev = CudaDevice::new(0).expect("CUDA device");
        let gate_host = [-4.0f32, -1.0, 0.0, 0.5, 2.0, 5.0];
        let up_host = [1.5f32, -2.0, 3.0, -4.0, 0.25, 0.75];
        let gate = dev.alloc(std::mem::size_of_val(&gate_host)).unwrap();
        let up = dev.alloc(std::mem::size_of_val(&up_host)).unwrap();
        let dst = dev.alloc(std::mem::size_of_val(&up_host)).unwrap();
        dev.copy_h2d(&gate, &gate_host).unwrap();
        dev.copy_h2d(&up, &up_host).unwrap();
        check(
            unsafe {
                gp_qwen_swiglu(
                    gate.as_f32(),
                    up.as_f32(),
                    dst.as_f32(),
                    gate_host.len() as i32,
                    dev.stream(),
                )
            },
            "cuda qwen swiglu",
        )
        .unwrap();
        let mut swiglu_out = [0.0f32; 6];
        dev.copy_d2h(&mut swiglu_out, &dst).unwrap();
        for (idx, ((actual, gate), up)) in swiglu_out.iter().zip(gate_host).zip(up_host).enumerate()
        {
            let expected = silu(gate) * up;
            assert!(
                (*actual - expected).abs() <= 2.0e-6,
                "swiglu idx={idx} actual={actual} expected={expected}"
            );
        }

        let mut values_host = up_host;
        let values = dev.alloc(std::mem::size_of_val(&values_host)).unwrap();
        dev.copy_h2d(&values, &values_host).unwrap();
        check(
            unsafe {
                gp_qwen_apply_silu_gate(
                    values.as_f32(),
                    gate.as_f32(),
                    values_host.len() as i32,
                    dev.stream(),
                )
            },
            "cuda qwen apply_silu_gate",
        )
        .unwrap();
        dev.copy_d2h(&mut values_host, &values).unwrap();
        for (idx, ((actual, gate), up)) in
            values_host.iter().zip(gate_host).zip(up_host).enumerate()
        {
            let expected = up * silu(gate);
            assert!(
                (*actual - expected).abs() <= 2.0e-6,
                "gate idx={idx} actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn qwen_argmax_selects_highest_logit_on_cuda() {
        let dev = CudaDevice::new(0).expect("CUDA device");
        let mut logits = vec![-1.0f32; 777];
        logits[19] = 4.0;
        logits[511] = 8.0;
        logits[701] = 8.0;
        logits[512] = f32::NAN;
        let logits_dev = dev.upload_bytes(as_bytes(&logits)).expect("upload logits");
        let token = dev.alloc(std::mem::size_of::<u32>()).expect("alloc token");
        let blocks = logits.len().div_ceil(256);
        let block_values = dev
            .alloc(blocks * std::mem::size_of::<f32>())
            .expect("alloc argmax values");
        let block_indices = dev
            .alloc(blocks * std::mem::size_of::<u32>())
            .expect("alloc argmax indices");
        check(
            unsafe {
                gp_qwen_argmax(
                    logits_dev.as_f32(),
                    token.as_u32(),
                    logits.len() as i32,
                    block_values.as_f32(),
                    block_indices.as_u32(),
                    blocks as i32,
                    dev.stream(),
                )
            },
            "cuda qwen argmax",
        )
        .unwrap();
        let mut out = [0_u32; 1];
        dev.copy_d2h(&mut out, &token).unwrap();
        assert_eq!(out[0], 511);
    }

    #[test]
    fn qwen_deltanet_decode_matches_cpu_on_cuda() {
        let dev = CudaDevice::new(0).expect("CUDA device");
        let heads = 2usize;
        let head_dim = 4usize;
        let q_host = patterned(heads * head_dim, 0.13);
        let k_host = patterned(heads * head_dim, -0.17);
        let v_host = patterned(heads * head_dim, 0.31);
        let beta_host = [0.25f32, -0.75];
        let alpha_host = [0.5f32, -1.25];
        let a_log_host = [-0.2f32, 0.4];
        let dt_bias_host = [0.1f32, -0.3];
        let mut state_host = patterned(heads * head_dim * head_dim, 0.07);
        let mut expected_state = state_host.clone();
        let expected = cpu_deltanet_decode(
            &q_host,
            &k_host,
            &v_host,
            &beta_host,
            &alpha_host,
            &a_log_host,
            &dt_bias_host,
            &mut expected_state,
            heads,
            head_dim,
        );

        let q = dev.upload_bytes(as_bytes(&q_host)).expect("upload q");
        let k = dev.upload_bytes(as_bytes(&k_host)).expect("upload k");
        let v = dev.upload_bytes(as_bytes(&v_host)).expect("upload v");
        let beta = dev.upload_bytes(as_bytes(&beta_host)).expect("upload beta");
        let alpha = dev
            .upload_bytes(as_bytes(&alpha_host))
            .expect("upload alpha");
        let a_log = dev
            .upload_bytes(as_bytes(&a_log_host))
            .expect("upload a_log");
        let dt_bias = dev
            .upload_bytes(as_bytes(&dt_bias_host))
            .expect("upload dt_bias");
        let state = dev
            .upload_bytes(as_bytes(&state_host))
            .expect("upload state");
        let out = dev
            .alloc(heads * head_dim * std::mem::size_of::<f32>())
            .expect("alloc out");
        dev.memset(&out, 0).unwrap();
        check(
            unsafe {
                gp_qwen_deltanet_decode(
                    q.as_f32(),
                    k.as_f32(),
                    v.as_f32(),
                    beta.as_f32(),
                    alpha.as_f32(),
                    a_log.as_f32(),
                    dt_bias.as_f32(),
                    state.as_f32(),
                    out.as_f32(),
                    heads as i32,
                    head_dim as i32,
                    dev.stream(),
                )
            },
            "cuda qwen deltanet_decode",
        )
        .unwrap();
        let mut actual = vec![0.0f32; heads * head_dim];
        dev.copy_d2h(&mut actual, &out).unwrap();
        dev.copy_d2h(&mut state_host, &state).unwrap();
        for (idx, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (*actual - *expected).abs() <= 2.0e-5,
                "out idx={idx} actual={actual} expected={expected}"
            );
        }
        for (idx, (actual, expected)) in state_host.iter().zip(&expected_state).enumerate() {
            assert!(
                (*actual - *expected).abs() <= 2.0e-5,
                "state idx={idx} actual={actual} expected={expected}"
            );
        }
    }

    fn cpu_causal_conv1d_silu(
        values: &mut [f32],
        weights: &[f32],
        state: &mut [f32],
        kernel: usize,
    ) {
        for channel in 0..values.len() {
            let base = channel * kernel;
            for i in 0..kernel - 1 {
                state[base + i] = state[base + i + 1];
            }
            state[base + kernel - 1] = values[channel];
            let mut acc = 0.0f32;
            for i in 0..kernel {
                acc += state[base + i] * weights[base + i];
            }
            values[channel] = acc / (1.0 + (-acc).exp());
        }
    }

    fn cpu_normalize_linear_qk(
        q: &mut [f32],
        k: &mut [f32],
        heads: usize,
        head_dim: usize,
        eps: f32,
    ) {
        let q_scale = 1.0 / (head_dim as f32).sqrt();
        for head in 0..heads {
            let base = head * head_dim;
            let qh = &mut q[base..base + head_dim];
            let kh = &mut k[base..base + head_dim];
            let q_norm = (qh.iter().map(|v| v * v).sum::<f32>() + eps).sqrt();
            let k_norm = (kh.iter().map(|v| v * v).sum::<f32>() + eps).sqrt();
            for v in qh {
                *v = *v / q_norm * q_scale;
            }
            for v in kh {
                *v /= k_norm;
            }
        }
    }

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn softplus(x: f32) -> f32 {
        if x > 20.0 {
            x
        } else {
            (1.0 + x.exp()).ln()
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_deltanet_decode(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        beta: &[f32],
        alpha: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
        state: &mut [f32],
        heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; heads * head_dim];
        for head in 0..heads {
            let base = head * head_dim;
            let beta_h = sigmoid(beta[head]);
            let decay = (-a_log[head].exp() * softplus(alpha[head] + dt_bias[head]))
                .exp()
                .clamp(0.0, 1.0);
            for value_idx in 0..head_dim {
                let row_base = (head * head_dim + value_idx) * head_dim;
                let mut prior = 0.0f32;
                for key_idx in 0..head_dim {
                    prior += state[row_base + key_idx] * k[base + key_idx];
                }
                let delta = (v[base + value_idx] - decay * prior) * beta_h;
                let mut attn = 0.0f32;
                for key_idx in 0..head_dim {
                    let idx = row_base + key_idx;
                    state[idx] = decay * state[idx] + k[base + key_idx] * delta;
                    attn += state[idx] * q[base + key_idx];
                }
                out[base + value_idx] = attn;
            }
        }
        out
    }

    fn patterned(len: usize, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 31 + 7) % 97) as f32 - 48.0) / 37.0 + offset)
            .collect()
    }

    fn as_bytes<T>(values: &[T]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
        }
    }
}

pub unsafe fn gp_rms_norm(
    src: *const f32,
    weight: *const f32,
    dst: *mut f32,
    rows: i32,
    dim: i32,
    eps: f32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe { (api.gp_rms_norm)(src, weight, dst, rows, dim, eps, stream) },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_rms_norm_add(
    src: *const f32,
    add: *const f32,
    weight: *const f32,
    dst: *mut f32,
    rows: i32,
    dim: i32,
    eps: f32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe { (api.gp_rms_norm_add)(src, add, weight, dst, rows, dim, eps, stream) },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_qwen_rms_norm(
    src: *const f32,
    weight: *const f32,
    dst: *mut f32,
    rows: i32,
    dim: i32,
    eps: f32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe { (api.gp_qwen_rms_norm)(src, weight, dst, rows, dim, eps, stream) },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_qwen_rms_norm_add(
    src: *const f32,
    add: *const f32,
    weight: *const f32,
    dst: *mut f32,
    rows: i32,
    dim: i32,
    eps: f32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.gp_qwen_rms_norm_add)(src, add, weight, dst, rows, dim, eps, stream)
        },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_qwen_add_rms_norm(
    lhs: *const f32,
    rhs: *const f32,
    weight: *const f32,
    sum_out: *mut f32,
    norm_out: *mut f32,
    rows: i32,
    dim: i32,
    eps: f32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.gp_qwen_add_rms_norm)(lhs, rhs, weight, sum_out, norm_out, rows, dim, eps, stream)
        },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_qwen_causal_conv1d_silu(
    values: *mut f32,
    weights: *const f32,
    state: *mut f32,
    channels: i32,
    kernel: i32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.gp_qwen_causal_conv1d_silu)(values, weights, state, channels, kernel, stream)
        },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

pub unsafe fn gp_qwen_normalize_linear_qk(
    q: *mut f32,
    k: *mut f32,
    heads: i32,
    head_dim: i32,
    eps: f32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe { (api.gp_qwen_normalize_linear_qk)(q, k, heads, head_dim, eps, stream) },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_qwen_deinterleave_q_gate(
    packed: *const f32,
    q_out: *mut f32,
    gate_out: *mut f32,
    rows: i32,
    heads: i32,
    head_dim: i32,
    packed_stride: i32,
    output_stride: i32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.gp_qwen_deinterleave_q_gate)(
                packed,
                q_out,
                gate_out,
                rows,
                heads,
                head_dim,
                packed_stride,
                output_stride,
                stream,
            )
        },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

pub unsafe fn gp_qwen_swiglu(
    gate: *const f32,
    up: *const f32,
    dst: *mut f32,
    total: i32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe { (api.gp_qwen_swiglu)(gate, up, dst, total, stream) },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

pub unsafe fn gp_qwen_apply_silu_gate(
    values: *mut f32,
    gate: *const f32,
    total: i32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe { (api.gp_qwen_apply_silu_gate)(values, gate, total, stream) },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

pub unsafe fn gp_qwen_apply_sigmoid_gate(
    values: *mut f32,
    gate: *const f32,
    total: i32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe { (api.gp_qwen_apply_sigmoid_gate)(values, gate, total, stream) },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

pub unsafe fn gp_qwen_add(
    lhs: *const f32,
    rhs: *const f32,
    dst: *mut f32,
    total: i32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe { (api.gp_qwen_add)(lhs, rhs, dst, total, stream) },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_qwen_argmax(
    logits: *const f32,
    token_out: *mut u32,
    total: i32,
    block_values: *mut f32,
    block_indices: *mut u32,
    block_count: i32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.gp_qwen_argmax)(
                logits,
                token_out,
                total,
                block_values,
                block_indices,
                block_count,
                stream,
            )
        },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_qwen_deltanet_decode(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    beta: *const f32,
    alpha: *const f32,
    a_log: *const f32,
    dt_bias: *const f32,
    state: *mut f32,
    out: *mut f32,
    heads: i32,
    head_dim: i32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.gp_qwen_deltanet_decode)(
                q, k, v, beta, alpha, a_log, dt_bias, state, out, heads, head_dim, stream,
            )
        },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_qwen_rope_decode(
    values: *mut f32,
    heads: i32,
    head_dim: i32,
    rope_dim: i32,
    position: i32,
    base_freq: f32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.gp_qwen_rope_decode)(
                values, heads, head_dim, rope_dim, position, base_freq, stream,
            )
        },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_qwen_cache_write(
    src: *const f32,
    cache: *mut f32,
    position: i32,
    heads: i32,
    head_dim: i32,
    max_context: i32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.gp_qwen_cache_write)(src, cache, position, heads, head_dim, max_context, stream)
        },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_qwen_attention_scores_decode(
    q: *const f32,
    k_cache: *const f32,
    scores: *mut f32,
    position: i32,
    q_heads: i32,
    kv_heads: i32,
    head_dim: i32,
    max_context: i32,
    scale: f32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.gp_qwen_attention_scores_decode)(
                q,
                k_cache,
                scores,
                position,
                q_heads,
                kv_heads,
                head_dim,
                max_context,
                scale,
                stream,
            )
        },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

pub unsafe fn gp_qwen_softmax_decode(
    scores: *mut f32,
    position: i32,
    heads: i32,
    max_context: i32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.gp_qwen_softmax_decode)(scores, position, heads, max_context, stream)
        },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_qwen_attention_values_decode(
    scores: *const f32,
    v_cache: *const f32,
    out: *mut f32,
    position: i32,
    q_heads: i32,
    kv_heads: i32,
    value_dim: i32,
    max_context: i32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.gp_qwen_attention_values_decode)(
                scores,
                v_cache,
                out,
                position,
                q_heads,
                kv_heads,
                value_dim,
                max_context,
                stream,
            )
        },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_rms_norm_heads(
    src: *const f32,
    weight: *const f32,
    dst: *mut f32,
    batch: i32,
    seq: i32,
    heads: i32,
    head_dim: i32,
    row_width: i32,
    eps: f32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.gp_rms_norm_heads)(
                src, weight, dst, batch, seq, heads, head_dim, row_width, eps, stream,
            )
        },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_split_heads(
    src: *const f32,
    dst: *mut f32,
    batch: i32,
    seq: i32,
    heads: i32,
    head_dim: i32,
    row_width: i32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.gp_split_heads)(src, dst, batch, seq, heads, head_dim, row_width, stream)
        },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_rope_neox(
    src: *const f32,
    dst: *mut f32,
    batch: i32,
    seq: i32,
    heads: i32,
    head_dim: i32,
    base_freq: f32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.gp_rope_neox)(src, dst, batch, seq, heads, head_dim, base_freq, stream)
        },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_attention_scores(
    blas: *mut c_void,
    q: *const f32,
    k: *const f32,
    scores: *mut f32,
    batch: i32,
    heads: i32,
    seq: i32,
    head_dim: i32,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.gp_attention_scores)(blas, q, k, scores, batch, heads, seq, head_dim)
        },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_softmax_mask(
    scores: *mut f32,
    mask: *const u32,
    batch: i32,
    heads: i32,
    seq: i32,
    sliding_window: i32,
    scale: f32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.gp_softmax_mask)(
                scores,
                mask,
                batch,
                heads,
                seq,
                sliding_window,
                scale,
                stream,
            )
        },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_attention_values(
    blas: *mut c_void,
    scores: *const f32,
    v: *const f32,
    out: *mut f32,
    batch: i32,
    heads: i32,
    seq: i32,
    head_dim: i32,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.gp_attention_values)(blas, scores, v, out, batch, heads, seq, head_dim)
        },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_merge_heads(
    src: *const f32,
    dst: *mut f32,
    batch: i32,
    seq: i32,
    heads: i32,
    head_dim: i32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe { (api.gp_merge_heads)(src, dst, batch, seq, heads, head_dim, stream) },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

pub unsafe fn gp_geglu(
    gate: *const f32,
    up: *const f32,
    dst: *mut f32,
    total: i32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe { (api.gp_geglu)(gate, up, dst, total, stream) },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_mean_pool(
    hidden: *const f32,
    mask: *const u32,
    dst: *mut f32,
    batch: i32,
    seq: i32,
    hidden_dim: i32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe { (api.gp_mean_pool)(hidden, mask, dst, batch, seq, hidden_dim, stream) },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

pub unsafe fn gp_l2_norm(
    src: *const f32,
    dst: *mut f32,
    rows: i32,
    dim: i32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe { (api.gp_l2_norm)(src, dst, rows, dim, stream) },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_mmq_matmul(
    dtype: i32,
    weights: *const c_void,
    src: *const f32,
    dst: *mut f32,
    q8_scratch: *mut c_void,
    fixup_scratch: *mut c_void,
    ncols_x: i64,
    stride_row_x: i64,
    nrows_x: i64,
    ncols_dst: i64,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.gp_mmq_matmul)(
                dtype,
                weights,
                src,
                dst,
                q8_scratch,
                fixup_scratch,
                ncols_x,
                stride_row_x,
                nrows_x,
                ncols_dst,
                stream,
            )
        },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_mmvq_matvec(
    dtype: i32,
    weights: *const c_void,
    src: *const f32,
    dst: *mut f32,
    q8_scratch: *mut c_void,
    ncols_x: i64,
    stride_row_x: i64,
    nrows_x: i64,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.gp_mmvq_matvec)(
                dtype,
                weights,
                src,
                dst,
                q8_scratch,
                ncols_x,
                stride_row_x,
                nrows_x,
                stream,
            )
        },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}
