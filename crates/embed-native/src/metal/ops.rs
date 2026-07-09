//! Byte-exact Rust ports of the `ggml_metal_op_*` dispatch functions
//! from `llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp` and the
//! companion `ggml_metal_library_get_pipeline_*` helpers from
//! `ggml-metal-device.cpp`.
//!
//! # Contract
//!
//! Each port produces the **same kernel-level behaviour** as the C++
//! original: identical pipeline name selection, identical
//! `ggml_metal_kargs_*` struct content, identical encoder argument
//! binding order, identical grid + threadgroup sizing.
//!
//! The Rust-side API differs from the C++ original in one way only:
//! instead of taking a `ggml_tensor*` and reading `ne[]`/`nb[]` off
//! it, each port takes explicit tensor-shape arguments (`ne00..ne03`,
//! `nb00..nb03`, etc.). That keeps the Rust side free of a ported
//! `ggml_tensor` graph-allocator layer (which is a separate port
//! item) — callers compute strides from their own buffer geometry
//! and pass them in directly. The dispatched kernel is still the
//! byte-exact vendored `ggml-metal.metal` one and reads the same
//! struct layout.
//!
//! ref:
//!   - vendor/metal/shaders/ggml/ggml-metal-impl.h
//!   - (upstream) ggml-metal-ops.cpp
//!   - (upstream) ggml-metal-device.cpp
//! (llama.cpp commit pinned in vendor/metal/ggml-metal.version)

use crate::metal::errors::set_last_error;
use crate::metal::ffi::{Buffer, ComputeEncoder, Device};
use crate::metal::kargs;

// ref: vendor/metal/shaders/ggml/ggml-metal-impl.h:75-90
// function-constant ID offsets — each kernel family reserves a
// contiguous block of 100 IDs starting at the offset below.
pub const FC_FLASH_ATTN_EXT_PAD: u32 = 100;
pub const FC_FLASH_ATTN_EXT_BLK: u32 = 200;
pub const FC_FLASH_ATTN_EXT: u32 = 300;
pub const FC_FLASH_ATTN_EXT_VEC: u32 = 400;
pub const FC_FLASH_ATTN_EXT_VEC_REDUCE: u32 = 500;
pub const FC_MUL_MV: u32 = 600;
pub const FC_MUL_MM: u32 = 700;
pub const FC_ROPE: u32 = 800;
pub const FC_SSM_CONV: u32 = 900;
pub const FC_SOLVE_TRI: u32 = 1000;
pub const FC_UNARY: u32 = 1200;
pub const FC_BIN: u32 = 1300;
pub const FC_GATED_DELTA_NET: u32 = 1600;

/// Helper: set a `bool` function-constant on an `MTLFunctionConstantValues`.
/// Matches `ggml_metal_cv_set_bool` in ggml-metal-device.m.
pub fn cv_set_bool(cv: &objc2_metal::MTLFunctionConstantValues, value: bool, index: u32) {
    let raw: u8 = if value { 1 } else { 0 };
    unsafe {
        cv.setConstantValue_type_atIndex(
            std::ptr::NonNull::new_unchecked(&raw as *const u8 as *mut std::ffi::c_void),
            objc2_metal::MTLDataType::Bool,
            index as objc2_foundation::NSUInteger,
        );
    }
}

/// Helper: set an `int16` function-constant. Matches
/// `ggml_metal_cv_set_int16` in ggml-metal-device.m.
pub fn cv_set_int16(cv: &objc2_metal::MTLFunctionConstantValues, value: i16, index: u32) {
    unsafe {
        cv.setConstantValue_type_atIndex(
            std::ptr::NonNull::new_unchecked(&value as *const i16 as *mut std::ffi::c_void),
            objc2_metal::MTLDataType::Short,
            index as objc2_foundation::NSUInteger,
        );
    }
}

/// Helper: set an `int32` function-constant. Matches the case in
/// `ggml-metal-device.m` where a declared FC is `int` (32-bit) rather
/// than `short` (16-bit). Dflash kernels declare their shape FCs as
/// `int` (see `vendor/metal/shaders/dflash/common.h`).
pub fn cv_set_int32(cv: &objc2_metal::MTLFunctionConstantValues, value: i32, index: u32) {
    unsafe {
        cv.setConstantValue_type_atIndex(
            std::ptr::NonNull::new_unchecked(&value as *const i32 as *mut std::ffi::c_void),
            objc2_metal::MTLDataType::Int,
            index as objc2_foundation::NSUInteger,
        );
    }
}

/// Public accessor for the `common::errors::last_error()` slot.
/// Smoke tests + examples can read the last `set_last_error(...)`
/// string without importing the private path.
pub fn last_error_str() -> String {
    crate::metal::errors::last_error()
}

/// Types mirroring `enum ggml_type` values used by the pipelines we
/// route to. Only the subset actually needed for Qwen3.5 forward is
/// listed. Keep numerical values aligned with `ggml.h::ggml_type` so
/// buffer-level ints match between Rust and the vendored header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q5_0 = 6,
    Q8_0 = 8,
    Bf16 = 30,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
}

impl GgmlType {
    /// ref: ggml.c::ggml_type_name — we only name the types we actually
    /// dispatch to, and the strings are the stable `kernel_*_<type>`
    /// fragment.
    pub fn name(self) -> &'static str {
        match self {
            GgmlType::F32 => "f32",
            GgmlType::F16 => "f16",
            GgmlType::Q5_0 => "q5_0",
            GgmlType::Q8_0 => "q8_0",
            GgmlType::Bf16 => "bf16",
            GgmlType::Q4_K => "q4_K",
            GgmlType::Q5_K => "q5_K",
            GgmlType::Q6_K => "q6_K",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Pipeline-selection helpers (ports from ggml-metal-device.cpp)
// ═══════════════════════════════════════════════════════════════════

/// ref: ggml-metal-device.cpp:84-98 — `ggml_metal_library_get_pipeline_cpy`
///
/// ```c
/// snprintf(base, 256, "kernel_cpy_%s_%s", ggml_type_name(tsrc), ggml_type_name(tdst));
/// ```
pub fn pipeline_name_cpy(tsrc: GgmlType, tdst: GgmlType) -> String {
    format!("kernel_cpy_{}_{}", tsrc.name(), tdst.name())
}

/// ref: ggml-metal-device.cpp:149-162 — `ggml_metal_library_get_pipeline_get_rows`
pub fn pipeline_name_get_rows(tsrc: GgmlType) -> String {
    format!("kernel_get_rows_{}", tsrc.name())
}

/// ref: ggml-metal-device.cpp:1099-1118 — `ggml_metal_library_get_pipeline_argmax`
pub fn pipeline_name_argmax(tsrc: GgmlType) -> String {
    format!("kernel_argmax_{}", tsrc.name())
}

/// ref: ggml-metal-device.cpp:1593-1634 — `ggml_metal_library_get_pipeline_norm`.
/// The C++ helper picks a name based on `n_fuse`:
///   n_fuse = 1 → `kernel_rms_norm_<type>`
///   n_fuse = 2 → `kernel_rms_norm_mul_<type>`
///   n_fuse = 3 → `kernel_rms_norm_mul_add_<type>`
/// and appends `_4` iff `ne00 % 4 == 0`.
pub fn pipeline_name_rms_norm(tsrc: GgmlType, n_fuse: i32, vec4: bool) -> Option<String> {
    let stem = match n_fuse {
        1 => "rms_norm",
        2 => "rms_norm_mul",
        3 => "rms_norm_mul_add",
        _ => {
            set_last_error(format!(
                "pipeline_name_rms_norm: n_fuse must be 1, 2, or 3, got {n_fuse}"
            ));
            return None;
        }
    };
    let suffix = if vec4 { "_4" } else { "" };
    Some(format!("kernel_{stem}_{}{suffix}", tsrc.name()))
}

/// ref: ggml-metal-device.cpp:1547 — `ggml_metal_library_get_pipeline_l2_norm`
pub fn pipeline_name_l2_norm(tsrc: GgmlType) -> String {
    format!("kernel_l2_norm_{}_{}", tsrc.name(), tsrc.name())
}

/// ref: ggml-metal-device.cpp:435 — `ggml_metal_library_get_pipeline_soft_max`
pub fn pipeline_name_soft_max(tsrc: GgmlType, vec4: bool) -> String {
    let suffix = if vec4 { "_4" } else { "" };
    format!("kernel_soft_max_{}{suffix}", tsrc.name())
}

/// ref: ggml-metal-device.cpp:265 — `ggml_metal_library_get_pipeline_unary`
pub fn pipeline_name_unary(tsrc: GgmlType, tdst: GgmlType, vec4: bool) -> String {
    let suffix = if vec4 { "_4" } else { "" };
    format!("kernel_unary_{}_{}{suffix}", tsrc.name(), tdst.name())
}

/// ref: ggml-metal-device.cpp:1636 — `ggml_metal_library_get_pipeline_rope`.
/// llama.cpp picks one of four RoPE variants per tensor (neox / norm /
/// multi / vision). For Qwen3.5 we hard-code `multi` since that's the
/// M-RoPE variant the model uses.
pub fn pipeline_name_rope_multi(tsrc: GgmlType) -> String {
    format!("kernel_rope_multi_{}", tsrc.name())
}

/// ref: ggml-metal-device.cpp:1636 — Gemma uses NeoX split-half RoPE.
pub fn pipeline_name_rope_neox(tsrc: GgmlType) -> String {
    format!("kernel_rope_neox_{}", tsrc.name())
}

/// Pipeline shape parameters returned alongside the pipeline object.
/// Mirrors `ggml_metal_pipeline_with_params::{nsg, nr0, nr1, smem}`
/// from `ggml-metal-device.h` so the op-dispatch sites can read them
/// off like the C++ original does.
#[derive(Clone, Copy, Debug)]
pub struct PipelineShape {
    pub nsg: i32,
    pub nr0: i32,
    pub nr1: i32,
    pub smem: usize,
}

/// Byte-exact port of the quant-type-specific N_R0/N_SG constants
/// from `vendor/metal/shaders/ggml/ggml-metal-impl.h:13-72` for the
/// mul_mv dispatch. Used inside `pipeline_mul_mv`.
pub fn mul_mv_shape(tsrc0: GgmlType, ne00: i32) -> (PipelineShape, &'static str) {
    match tsrc0 {
        GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16 => {
            if ne00 < 32 {
                (
                    PipelineShape {
                        nsg: 1,
                        nr0: 32,
                        nr1: 1,
                        smem: 0,
                    },
                    "_short",
                )
            } else {
                let nsg = 4.min((ne00 + 127) / 128);
                let nr0 = 2;
                let smem = 32 * 4 * (nr0 as usize);
                let suffix = if ne00 % 4 == 0 { "_4" } else { "" };
                (
                    PipelineShape {
                        nsg,
                        nr0,
                        nr1: 1,
                        smem,
                    },
                    suffix,
                )
            }
        }
        // ref: ggml-metal-impl.h:20-21 (N_R0_Q5_0 = 4, N_SG_Q5_0 = 2)
        GgmlType::Q5_0 => (
            PipelineShape {
                nsg: 2,
                nr0: 4,
                nr1: 1,
                smem: 0,
            },
            "",
        ),
        // ref: ggml-metal-impl.h:26-27 (N_R0_Q8_0 = 2, N_SG_Q8_0 = 4)
        GgmlType::Q8_0 => (
            PipelineShape {
                nsg: 4,
                nr0: 2,
                nr1: 1,
                smem: 32 * 4 * 2,
            },
            "",
        ),
        // ref: ggml-metal-impl.h:37-38 (N_R0_Q4_K = 2, N_SG_Q4_K = 2)
        GgmlType::Q4_K => (
            PipelineShape {
                nsg: 2,
                nr0: 2,
                nr1: 1,
                smem: 0,
            },
            "",
        ),
        // ref: ggml-metal-impl.h:54-55 (N_R0_Q5_K = 1, N_SG_Q5_K = 2)
        GgmlType::Q5_K => (
            PipelineShape {
                nsg: 2,
                nr0: 1,
                nr1: 1,
                smem: 0,
            },
            "",
        ),
        // ref: ggml-metal-impl.h:44-45 (N_R0_Q6_K = 2, N_SG_Q6_K = 2)
        GgmlType::Q6_K => (
            PipelineShape {
                nsg: 2,
                nr0: 2,
                nr1: 1,
                smem: 0,
            },
            "",
        ),
    }
}

/// ref: ggml-metal-device.cpp:703-... `ggml_metal_library_get_pipeline_mul_mv`
/// — returns the kernel name + shape params given src types.
pub fn pipeline_name_mul_mv(
    tsrc0: GgmlType,
    tsrc1: GgmlType,
    ne00: i32,
) -> (String, PipelineShape) {
    let (shape, suffix) = mul_mv_shape(tsrc0, ne00);
    let name = format!("kernel_mul_mv_{}_{}{suffix}", tsrc0.name(), tsrc1.name());
    (name, shape)
}

/// ref: ggml-metal-device.cpp:1514 — `ggml_metal_library_get_pipeline_bin_one`
/// for the single-operator binary path (non-fused add/mul).
pub fn pipeline_name_bin(ggml_op: GgmlBinOp) -> String {
    let _ = ggml_op;
    "kernel_bin_fuse_f32_f32_f32".to_string()
}

fn bin_op_num(ggml_op: GgmlBinOp) -> i16 {
    match ggml_op {
        GgmlBinOp::Add => 0,
        GgmlBinOp::Sub => 1,
        GgmlBinOp::Mul => 2,
        GgmlBinOp::Div => 3,
    }
}

#[derive(Clone, Copy, Debug)]
pub enum GgmlBinOp {
    Add,
    Mul,
    Div,
    Sub,
}

#[derive(Clone, Copy, Debug)]
pub enum GgmlGluOp {
    Geglu,
}

// ═══════════════════════════════════════════════════════════════════
//  ggml_metal_op_* dispatch ports
// ═══════════════════════════════════════════════════════════════════

/// Byte-exact port of `ggml_metal_op_cpy` (ggml-metal-ops.cpp lines
/// 1844-... main body). The `inplace` path is elided — for the cpy
/// use-cases the Rust caller controls whether a copy is required, so
/// we always invoke the separate-kernel path (same code as the
/// C++ `if (!inplace)` branch). Callers that need the fused in-place
/// add-copy semantics dispatch `op_bin` directly.
///
/// ref: ggml-metal-ops.cpp:1844 + inner body at lines 643-683
#[allow(clippy::too_many_arguments)]
pub fn op_cpy(
    enc: &ComputeEncoder,
    dev: &Device,
    src: &Buffer,
    dst: &Buffer,
    tsrc: GgmlType,
    tdst: GgmlType,
    // tensor dims for src (ne00..ne03 = sizes, nb00..nb03 = strides in bytes)
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne03: i64,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    // tensor dims for dst (ne0..ne3, nb0..nb3)
    ne0: i64,
    ne1: i64,
    ne2: i64,
    ne3: i64,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
) -> bool {
    let name = pipeline_name_cpy(tsrc, tdst);
    let Some(pso) = dev.pipeline(&name) else {
        set_last_error(format!("op_cpy: pipeline `{name}` not found in metallib"));
        return false;
    };

    // Byte-exact struct fill from ggml-metal-ops.cpp:653-671.
    // Note: the cpp fills `nk0 = ne00` (first field), then ne00..03,
    // nb00..03, ne0..3, nb0..3. We mirror exactly.
    let args = kargs::KargsCpy {
        nk0: ne00,
        ne00,
        ne01,
        ne02,
        ne03,
        nb00,
        nb01,
        nb02,
        nb03,
        ne0,
        ne1,
        ne2,
        ne3,
        nb0,
        nb1,
        nb2,
        nb3,
    };

    // Encoder setup — same buffer-slot ordering as the C++ original:
    //   slot 0: kargs bytes
    //   slot 1: src0 (source tensor)
    //   slot 2: dst  (destination tensor)
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, src, 0);
    enc.set_buffer(2, dst, 0);

    // Grid + threadgroup: `ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1)`
    //   where `nth = min(pipeline_max_threads_per_tg, ne00)`.
    //
    // We don't have a `pipeline_max_threads` accessor on our Device
    // today — MTLComputePipelineState exposes it via
    // `maxTotalThreadsPerThreadgroup`. For Metal 3+ this is at least
    // 1024 on Apple Silicon, so clamping to `ne00` (which is bounded
    // by the model's hidden dim / head dim) is safe in practice. The
    // exact lookup via objc2-metal is a small follow-up — tracking
    // the proper port of `ggml_metal_pipeline_max_theads_per_threadgroup`.
    const FALLBACK_NTH: i64 = 1024;
    let nth = ne00.min(FALLBACK_NTH) as usize;

    enc.dispatch_threadgroups((ne01 as usize, ne02 as usize, ne03 as usize), (nth, 1, 1));

    true
}

/// Byte-exact port of the single-op (non-fused) `ggml_metal_op_bin`
/// path (ggml-metal-ops.cpp:3056-3211). For `n_fuse > 1` the C++
/// original chains adds/muls across subsequent graph nodes; the Rust
/// API exposes only the common single-op case since our graph driver
/// emits one op per call.
///
/// Handles `kernel_add_fuse_impl`-style dispatch: kargs + three
/// buffers (src0, src1, dst).
#[allow(clippy::too_many_arguments)]
pub fn op_bin(
    enc: &ComputeEncoder,
    dev: &Device,
    op_kind: GgmlBinOp,
    src0: &Buffer,
    src1: &Buffer,
    dst: &Buffer,
    // src0 dims
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    // src1 dims
    ne10: i32,
    ne11: i32,
    ne12: i32,
    ne13: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    // dst dims
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
) -> bool {
    let name = pipeline_name_bin(op_kind);
    let cache_key = format!("{name}_op={}_nf=1", bin_op_num(op_kind));
    let Some(pso) = dev.pipeline_with_constants(&cache_key, &name, |cv| {
        cv_set_int16(cv, bin_op_num(op_kind), FC_BIN + 0);
        cv_set_int16(cv, 1, FC_BIN + 1);
        cv_set_bool(cv, false, FC_BIN + 2);
    }) else {
        set_last_error(format!("op_bin: pipeline `{name}` not found in metallib"));
        return false;
    };

    // Byte-exact struct fill from ggml-metal-ops.cpp:3084-3109.
    let args = kargs::KargsBin {
        ne00,
        ne01,
        ne02,
        ne03,
        nb00,
        nb01,
        nb02,
        nb03,
        ne10,
        ne11,
        ne12,
        ne13,
        nb10,
        nb11,
        nb12,
        nb13,
        ne0,
        ne1,
        ne2,
        ne3,
        nb0,
        nb1,
        nb2,
        nb3,
        offs: 0,
        o1: [0; 8],
    };

    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, src0, 0);
    enc.set_buffer(2, src1, 0);
    enc.set_buffer(3, dst, 0);

    // Grid: ref ggml-metal-ops.cpp:3201-3209 — non-c4, non-cnt path:
    //   nth_max = min(256, pipeline_max_threads); nth doubles while
    //   2*nth < args.ne0 && nth < nth_max; dispatch(ne01, ne02, ne03, nth, 1, 1).
    const NTH_MAX: i64 = 256;
    let mut nth: i64 = 1;
    while 2 * nth < ne0 as i64 && nth < NTH_MAX {
        nth *= 2;
    }
    enc.dispatch_threadgroups(
        (ne01 as usize, ne02 as usize, ne03 as usize),
        (nth as usize, 1, 1),
    );
    true
}

/// Byte-exact port of `ggml_metal_op_unary` (ggml-metal-ops.cpp:735-826).
/// Covers silu / gelu / relu / sigmoid / swish / clamp / scale / fill
/// etc. The `UnaryParams` struct carries the optional constants
/// (slope / scale / bias / val / min / max) that specific unary
/// kernels consume; for silu/sigmoid the caller passes `Default::default`.
#[derive(Default, Clone, Copy)]
pub struct UnaryParams {
    pub slope: f32,
    pub scale: f32,
    pub bias: f32,
    pub val: f32,
    pub min: f32,
    pub max: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QwenNormArgs {
    rows: i32,
    dim: i32,
    eps: f32,
    qwen_scale: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QwenAddArgs {
    total: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QwenConvArgs {
    channels: i32,
    k_width: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QwenConvRowsArgs {
    rows: i32,
    channels: i32,
    k_width: i32,
    pad: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QwenHeadsArgs {
    heads: i32,
    head_dim: i32,
    eps: f32,
    pad: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QwenHeadsRowsArgs {
    rows: i32,
    heads: i32,
    head_dim: i32,
    q_stride: i32,
    k_stride: i32,
    eps: f32,
    pad0: i32,
    pad1: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QwenDeltaNetArgs {
    heads: i32,
    head_dim: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QwenDeltaNetRowsArgs {
    rows: i32,
    heads: i32,
    head_dim: i32,
    q_stride: i32,
    k_stride: i32,
    v_stride: i32,
    beta_stride: i32,
    alpha_stride: i32,
    out_stride: i32,
    pad0: i32,
    pad1: i32,
    pad2: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QwenRopeArgs {
    heads: i32,
    head_dim: i32,
    rope_dim: i32,
    position: i32,
    base_freq: f32,
    pad0: i32,
    pad1: i32,
    pad2: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QwenRopeRowsArgs {
    rows: i32,
    heads: i32,
    head_dim: i32,
    rope_dim: i32,
    position: i32,
    stride: i32,
    base_freq: f32,
    pad: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QwenCacheArgs {
    position: i32,
    heads: i32,
    head_dim: i32,
    max_context: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QwenCacheRowsArgs {
    rows: i32,
    position: i32,
    heads: i32,
    head_dim: i32,
    max_context: i32,
    src_stride: i32,
    pad0: i32,
    pad1: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QwenAttentionArgs {
    position: i32,
    q_heads: i32,
    kv_heads: i32,
    dim: i32,
    max_context: i32,
    scale: f32,
    pad0: i32,
    pad1: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QwenAttentionRowsArgs {
    rows: i32,
    position: i32,
    q_heads: i32,
    kv_heads: i32,
    dim: i32,
    max_context: i32,
    q_stride: i32,
    score_stride: i32,
    scale: f32,
    pad0: i32,
    pad1: i32,
    pad2: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QwenGateRowsArgs {
    rows: i32,
    width: i32,
    value_stride: i32,
    gate_stride: i32,
}

#[derive(Clone, Copy, Debug)]
pub enum GgmlUnaryOp {
    Scale,
    Gelu,
    Tanh,
    Relu,
    Sigmoid,
}

fn unary_op_num(op: GgmlUnaryOp) -> i16 {
    match op {
        GgmlUnaryOp::Scale => 10,
        GgmlUnaryOp::Tanh => 100,
        GgmlUnaryOp::Relu => 101,
        GgmlUnaryOp::Sigmoid => 102,
        GgmlUnaryOp::Gelu => 103,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn op_unary(
    enc: &ComputeEncoder,
    dev: &Device,
    op_kind: GgmlUnaryOp,
    tsrc: GgmlType,
    tdst: GgmlType,
    src: &Buffer,
    dst: &Buffer,
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    params: UnaryParams,
) -> bool {
    let vec4 = ne00 % 4 == 0;
    let is_contiguous_small = nb00 == 4
        && nb0 == 4
        && nb01 == (ne00 as u64) * 4
        && nb1 == (ne0 as u64) * 4
        && (ne00 as i64 * ne01 as i64 * ne02 as i64 * ne03 as i64) < 32768;
    let pipeline_name = pipeline_name_unary(tsrc, tdst, vec4);
    let cache_key = format!(
        "{pipeline_name}_op={}_cnt={}",
        unary_op_num(op_kind),
        is_contiguous_small as i32
    );
    let Some(pso) = dev.pipeline_with_constants(&cache_key, &pipeline_name, |cv| {
        cv_set_int16(cv, unary_op_num(op_kind), FC_UNARY + 0);
        cv_set_bool(cv, is_contiguous_small, FC_UNARY + 1);
    }) else {
        set_last_error(format!(
            "op_unary: pipeline `{pipeline_name}` not found in metallib"
        ));
        return false;
    };

    // Byte-exact struct fill from ggml-metal-ops.cpp:751-776.
    let k_ne00 = if vec4 { ne00 / 4 } else { ne00 };
    let k_ne0 = if vec4 { ne0 / 4 } else { ne0 };
    let args = kargs::KargsUnary {
        ne00: k_ne00,
        ne01,
        ne02,
        ne03,
        nb00,
        nb01,
        nb02,
        nb03,
        ne0: k_ne0,
        ne1,
        ne2,
        ne3,
        nb0,
        nb1,
        nb2,
        nb3,
        slope: params.slope,
        scale: params.scale,
        bias: params.bias,
        val: params.val,
        min: params.min,
        max: params.max,
    };

    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, src, 0);
    enc.set_buffer(2, dst, 0);

    if is_contiguous_small {
        let elems = (ne00 as i64 * ne01 as i64 * ne02 as i64 * ne03 as i64) as usize;
        let elems = if vec4 { elems / 4 } else { elems };
        enc.dispatch_threadgroups((elems, 1, 1), (1, 1, 1));
    } else {
        // Grid: ref ggml-metal-ops.cpp:817-824 — non-cnt path:
        //   nth = min(args.ne00, nth_max); nk0 = (args.ne00 + nth - 1)/nth;
        //   dispatch(nk0*ne01, ne02, ne03, nth, 1, 1).
        const NTH_MAX: i32 = 256;
        let nth = k_ne00.min(NTH_MAX);
        let nk0 = (k_ne00 + nth - 1) / nth;
        enc.dispatch_threadgroups(
            ((nk0 * ne01) as usize, ne02 as usize, ne03 as usize),
            (nth as usize, 1, 1),
        );
    }
    true
}

/// Byte-exact port of `ggml_metal_op_glu` for the two-input GEGLU
/// path: `dst = gelu(src0) * src1`.
#[allow(clippy::too_many_arguments)]
pub fn op_glu(
    enc: &ComputeEncoder,
    dev: &Device,
    op_kind: GgmlGluOp,
    src0: &Buffer,
    src1: &Buffer,
    dst: &Buffer,
    ne00: i32,
    nb01: u64,
    ne10: i32,
    nb11: u64,
    ne0: i32,
    nb1: u64,
    nrows: i32,
) -> bool {
    let name = match op_kind {
        GgmlGluOp::Geglu => "kernel_geglu_f32",
    };
    let Some(pso) = dev.pipeline(name) else {
        set_last_error(format!("op_glu: pipeline `{name}` not found"));
        return false;
    };
    let args = kargs::KargsGlu {
        ne00,
        nb01,
        ne10,
        nb11,
        ne0,
        nb1,
        i00: 0,
        i10: 0,
        alpha: 0.0,
        limit: 0.0,
    };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, src0, 0);
    enc.set_buffer(2, src1, 0);
    enc.set_buffer(3, dst, 0);
    let nth = (ne00 / 2).clamp(1, 1024);
    enc.dispatch_threadgroups((nrows as usize, 1, 1), (nth as usize, 1, 1));
    true
}

pub fn op_qwen_rms_norm_f32(
    enc: &ComputeEncoder,
    dev: &Device,
    src: &Buffer,
    weight: &Buffer,
    weight_offset: usize,
    dst: &Buffer,
    rows: i32,
    dim: i32,
    eps: f32,
    qwen_scale: bool,
) -> bool {
    let name = "embed_native_qwen_rms_norm_f32";
    let Some(pso) = dev.pipeline(name) else {
        set_last_error(format!("op_qwen_rms_norm_f32: pipeline `{name}` not found"));
        return false;
    };
    let args = QwenNormArgs {
        rows,
        dim,
        eps,
        qwen_scale: i32::from(qwen_scale),
    };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, src, 0);
    enc.set_buffer(2, weight, weight_offset);
    enc.set_buffer(3, dst, 0);
    enc.set_threadgroup_memory_size(256 * 4, 0);
    enc.dispatch_threadgroups((rows as usize, 1, 1), (256, 1, 1));
    true
}

#[allow(clippy::too_many_arguments)]
pub fn op_qwen_add_rms_norm_f32(
    enc: &ComputeEncoder,
    dev: &Device,
    lhs: &Buffer,
    rhs: &Buffer,
    weight: &Buffer,
    weight_offset: usize,
    sum_out: &Buffer,
    norm_out: &Buffer,
    rows: i32,
    dim: i32,
    eps: f32,
) -> bool {
    let name = "embed_native_qwen_add_rms_norm_f32";
    let Some(pso) = dev.pipeline(name) else {
        set_last_error(format!(
            "op_qwen_add_rms_norm_f32: pipeline `{name}` not found"
        ));
        return false;
    };
    let args = QwenNormArgs {
        rows,
        dim,
        eps,
        qwen_scale: 1,
    };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, lhs, 0);
    enc.set_buffer(2, rhs, 0);
    enc.set_buffer(3, weight, weight_offset);
    enc.set_buffer(4, sum_out, 0);
    enc.set_buffer(5, norm_out, 0);
    enc.set_threadgroup_memory_size(256 * 4, 0);
    enc.dispatch_threadgroups((rows as usize, 1, 1), (256, 1, 1));
    true
}

pub fn op_qwen_swiglu_f32(
    enc: &ComputeEncoder,
    dev: &Device,
    gate: &Buffer,
    up: &Buffer,
    dst: &Buffer,
    total: i32,
) -> bool {
    let name = "embed_native_qwen_swiglu_f32";
    let Some(pso) = dev.pipeline(name) else {
        set_last_error(format!("op_qwen_swiglu_f32: pipeline `{name}` not found"));
        return false;
    };
    let args = QwenAddArgs { total };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, gate, 0);
    enc.set_buffer(2, up, 0);
    enc.set_buffer(3, dst, 0);
    enc.dispatch(((total as usize).max(1), 1, 1), (256, 1, 1));
    true
}

pub fn op_qwen_apply_silu_gate_f32(
    enc: &ComputeEncoder,
    dev: &Device,
    values: &Buffer,
    values_offset: usize,
    gate: &Buffer,
    gate_offset: usize,
    total: i32,
) -> bool {
    let name = "embed_native_qwen_apply_silu_gate_f32";
    let Some(pso) = dev.pipeline(name) else {
        set_last_error(format!(
            "op_qwen_apply_silu_gate_f32: pipeline `{name}` not found"
        ));
        return false;
    };
    let args = QwenAddArgs { total };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, values, values_offset);
    enc.set_buffer(2, gate, gate_offset);
    enc.dispatch(((total as usize).max(1), 1, 1), (256, 1, 1));
    true
}

pub fn op_qwen_apply_sigmoid_gate_f32(
    enc: &ComputeEncoder,
    dev: &Device,
    values: &Buffer,
    values_offset: usize,
    gate: &Buffer,
    gate_offset: usize,
    total: i32,
) -> bool {
    let name = "embed_native_qwen_apply_sigmoid_gate_f32";
    let Some(pso) = dev.pipeline(name) else {
        set_last_error(format!(
            "op_qwen_apply_sigmoid_gate_f32: pipeline `{name}` not found"
        ));
        return false;
    };
    let args = QwenAddArgs { total };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, values, values_offset);
    enc.set_buffer(2, gate, gate_offset);
    enc.dispatch(((total as usize).max(1), 1, 1), (256, 1, 1));
    true
}

pub fn op_qwen_add_f32(
    enc: &ComputeEncoder,
    dev: &Device,
    lhs: &Buffer,
    rhs: &Buffer,
    dst: &Buffer,
    total: i32,
) -> bool {
    let name = "embed_native_qwen_add_f32";
    let Some(pso) = dev.pipeline(name) else {
        set_last_error(format!("op_qwen_add_f32: pipeline `{name}` not found"));
        return false;
    };
    let args = QwenAddArgs { total };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, lhs, 0);
    enc.set_buffer(2, rhs, 0);
    enc.set_buffer(3, dst, 0);
    enc.dispatch(((total as usize).max(1), 1, 1), (256, 1, 1));
    true
}

pub fn op_qwen_causal_conv1d_silu_f32(
    enc: &ComputeEncoder,
    dev: &Device,
    values: &Buffer,
    weight: &Buffer,
    weight_offset: usize,
    state: &Buffer,
    channels: i32,
    kernel: i32,
) -> bool {
    let name = "embed_native_qwen_causal_conv1d_silu_f32";
    let Some(pso) = dev.pipeline(name) else {
        set_last_error(format!(
            "op_qwen_causal_conv1d_silu_f32: pipeline `{name}` not found"
        ));
        return false;
    };
    let args = QwenConvArgs {
        channels,
        k_width: kernel,
    };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, values, 0);
    enc.set_buffer(2, weight, weight_offset);
    enc.set_buffer(3, state, 0);
    enc.dispatch(((channels as usize).max(1), 1, 1), (256, 1, 1));
    true
}

pub fn op_qwen_causal_conv1d_silu_rows_f32(
    enc: &ComputeEncoder,
    dev: &Device,
    values: &Buffer,
    weight: &Buffer,
    weight_offset: usize,
    state: &Buffer,
    rows: i32,
    channels: i32,
    kernel: i32,
) -> bool {
    let name = "embed_native_qwen_causal_conv1d_silu_rows_f32";
    let Some(pso) = dev.pipeline(name) else {
        set_last_error(format!(
            "op_qwen_causal_conv1d_silu_rows_f32: pipeline `{name}` not found"
        ));
        return false;
    };
    let args = QwenConvRowsArgs {
        rows,
        channels,
        k_width: kernel,
        pad: 0,
    };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, values, 0);
    enc.set_buffer(2, weight, weight_offset);
    enc.set_buffer(3, state, 0);
    enc.dispatch(((channels as usize).max(1), 1, 1), (256, 1, 1));
    true
}

pub fn op_qwen_normalize_linear_qk_f32(
    enc: &ComputeEncoder,
    dev: &Device,
    q: &Buffer,
    q_offset: usize,
    k: &Buffer,
    k_offset: usize,
    heads: i32,
    head_dim: i32,
    eps: f32,
) -> bool {
    let name = "embed_native_qwen_normalize_linear_qk_f32";
    let Some(pso) = dev.pipeline(name) else {
        set_last_error(format!(
            "op_qwen_normalize_linear_qk_f32: pipeline `{name}` not found"
        ));
        return false;
    };
    let args = QwenHeadsArgs {
        heads,
        head_dim,
        eps,
        pad: 0,
    };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, q, q_offset);
    enc.set_buffer(2, k, k_offset);
    enc.set_threadgroup_memory_size(2 * 256 * 4, 0);
    enc.dispatch_threadgroups((heads as usize, 1, 1), (256, 1, 1));
    true
}

#[allow(clippy::too_many_arguments)]
pub fn op_qwen_normalize_linear_qk_rows_f32(
    enc: &ComputeEncoder,
    dev: &Device,
    q: &Buffer,
    q_offset: usize,
    k: &Buffer,
    k_offset: usize,
    rows: i32,
    heads: i32,
    head_dim: i32,
    q_stride: i32,
    k_stride: i32,
    eps: f32,
) -> bool {
    let name = "embed_native_qwen_normalize_linear_qk_rows_f32";
    let Some(pso) = dev.pipeline(name) else {
        set_last_error(format!(
            "op_qwen_normalize_linear_qk_rows_f32: pipeline `{name}` not found"
        ));
        return false;
    };
    let args = QwenHeadsRowsArgs {
        rows,
        heads,
        head_dim,
        q_stride,
        k_stride,
        eps,
        pad0: 0,
        pad1: 0,
    };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, q, q_offset);
    enc.set_buffer(2, k, k_offset);
    enc.set_threadgroup_memory_size(2 * 256 * 4, 0);
    enc.dispatch_threadgroups((rows as usize, heads as usize, 1), (256, 1, 1));
    true
}

#[allow(clippy::too_many_arguments)]
pub fn op_qwen_deinterleave_q_gate_rows_f32(
    enc: &ComputeEncoder,
    dev: &Device,
    packed: &Buffer,
    packed_offset: usize,
    q_out: &Buffer,
    q_out_offset: usize,
    gate_out: &Buffer,
    gate_out_offset: usize,
    rows: i32,
    heads: i32,
    head_dim: i32,
    packed_stride: i32,
    output_stride: i32,
) -> bool {
    let name = "embed_native_qwen_deinterleave_q_gate_rows_f32";
    let Some(pso) = dev.pipeline(name) else {
        set_last_error(format!(
            "op_qwen_deinterleave_q_gate_rows_f32: pipeline `{name}` not found"
        ));
        return false;
    };
    let args = QwenHeadsRowsArgs {
        rows,
        heads,
        head_dim,
        q_stride: packed_stride,
        k_stride: output_stride,
        eps: 0.0,
        pad0: 0,
        pad1: 0,
    };
    let total = rows * heads * head_dim;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, packed, packed_offset);
    enc.set_buffer(2, q_out, q_out_offset);
    enc.set_buffer(3, gate_out, gate_out_offset);
    enc.dispatch(((total as usize).max(1), 1, 1), (256, 1, 1));
    true
}

#[allow(clippy::too_many_arguments)]
pub fn op_qwen_rms_norm_strided_rows_f32(
    enc: &ComputeEncoder,
    dev: &Device,
    values: &Buffer,
    values_offset: usize,
    weight: &Buffer,
    weight_offset: usize,
    rows: i32,
    heads: i32,
    head_dim: i32,
    stride: i32,
    eps: f32,
) -> bool {
    let name = "embed_native_qwen_rms_norm_strided_rows_f32";
    let Some(pso) = dev.pipeline(name) else {
        set_last_error(format!(
            "op_qwen_rms_norm_strided_rows_f32: pipeline `{name}` not found"
        ));
        return false;
    };
    let args = QwenHeadsRowsArgs {
        rows,
        heads,
        head_dim,
        q_stride: stride,
        k_stride: 0,
        eps,
        pad0: 0,
        pad1: 0,
    };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, values, values_offset);
    enc.set_buffer(2, weight, weight_offset);
    enc.set_threadgroup_memory_size(256 * 4, 0);
    enc.dispatch_threadgroups((rows as usize, heads as usize, 1), (256, 1, 1));
    true
}

#[allow(clippy::too_many_arguments)]
pub fn op_qwen_deltanet_decode_f32(
    enc: &ComputeEncoder,
    dev: &Device,
    q: &Buffer,
    q_offset: usize,
    k: &Buffer,
    k_offset: usize,
    v: &Buffer,
    v_offset: usize,
    beta: &Buffer,
    alpha: &Buffer,
    a_log: &Buffer,
    a_log_offset: usize,
    dt_bias: &Buffer,
    dt_bias_offset: usize,
    state: &Buffer,
    out: &Buffer,
    heads: i32,
    head_dim: i32,
) -> bool {
    let name = "embed_native_qwen_deltanet_decode_f32";
    let Some(pso) = dev.pipeline(name) else {
        set_last_error(format!(
            "op_qwen_deltanet_decode_f32: pipeline `{name}` not found"
        ));
        return false;
    };
    let args = QwenDeltaNetArgs { heads, head_dim };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, q, q_offset);
    enc.set_buffer(2, k, k_offset);
    enc.set_buffer(3, v, v_offset);
    enc.set_buffer(4, beta, 0);
    enc.set_buffer(5, alpha, 0);
    enc.set_buffer(6, a_log, a_log_offset);
    enc.set_buffer(7, dt_bias, dt_bias_offset);
    enc.set_buffer(8, state, 0);
    enc.set_buffer(9, out, 0);
    enc.set_threadgroup_memory_size(2 * 256 * 4, 0);
    enc.dispatch_threadgroups((heads as usize, head_dim as usize, 1), (256, 1, 1));
    true
}

#[allow(clippy::too_many_arguments)]
pub fn op_qwen_deltanet_decode_rows_f32(
    enc: &ComputeEncoder,
    dev: &Device,
    q: &Buffer,
    q_offset: usize,
    k: &Buffer,
    k_offset: usize,
    v: &Buffer,
    v_offset: usize,
    beta: &Buffer,
    alpha: &Buffer,
    a_log: &Buffer,
    a_log_offset: usize,
    dt_bias: &Buffer,
    dt_bias_offset: usize,
    state: &Buffer,
    out: &Buffer,
    rows: i32,
    heads: i32,
    head_dim: i32,
    q_stride: i32,
    k_stride: i32,
    v_stride: i32,
    beta_stride: i32,
    alpha_stride: i32,
    out_stride: i32,
) -> bool {
    let name = "embed_native_qwen_deltanet_decode_rows_f32";
    let Some(pso) = dev.pipeline(name) else {
        set_last_error(format!(
            "op_qwen_deltanet_decode_rows_f32: pipeline `{name}` not found"
        ));
        return false;
    };
    let args = QwenDeltaNetRowsArgs {
        rows,
        heads,
        head_dim,
        q_stride,
        k_stride,
        v_stride,
        beta_stride,
        alpha_stride,
        out_stride,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, q, q_offset);
    enc.set_buffer(2, k, k_offset);
    enc.set_buffer(3, v, v_offset);
    enc.set_buffer(4, beta, 0);
    enc.set_buffer(5, alpha, 0);
    enc.set_buffer(6, a_log, a_log_offset);
    enc.set_buffer(7, dt_bias, dt_bias_offset);
    enc.set_buffer(8, state, 0);
    enc.set_buffer(9, out, 0);
    enc.set_threadgroup_memory_size(2 * 256 * 4, 0);
    enc.dispatch_threadgroups((heads as usize, head_dim as usize, 1), (256, 1, 1));
    true
}

pub fn op_qwen_rope_decode_f32(
    enc: &ComputeEncoder,
    dev: &Device,
    values: &Buffer,
    values_offset: usize,
    heads: i32,
    head_dim: i32,
    rope_dim: i32,
    position: i32,
    base_freq: f32,
) -> bool {
    let name = "embed_native_qwen_rope_decode_f32";
    let Some(pso) = dev.pipeline(name) else {
        set_last_error(format!(
            "op_qwen_rope_decode_f32: pipeline `{name}` not found"
        ));
        return false;
    };
    let args = QwenRopeArgs {
        heads,
        head_dim,
        rope_dim,
        position,
        base_freq,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    };
    let total = heads * (rope_dim / 2);
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, values, values_offset);
    enc.dispatch(((total as usize).max(1), 1, 1), (256, 1, 1));
    true
}

#[allow(clippy::too_many_arguments)]
pub fn op_qwen_rope_rows_f32(
    enc: &ComputeEncoder,
    dev: &Device,
    values: &Buffer,
    values_offset: usize,
    rows: i32,
    heads: i32,
    head_dim: i32,
    rope_dim: i32,
    position: i32,
    stride: i32,
    base_freq: f32,
) -> bool {
    let name = "embed_native_qwen_rope_rows_f32";
    let Some(pso) = dev.pipeline(name) else {
        set_last_error(format!(
            "op_qwen_rope_rows_f32: pipeline `{name}` not found"
        ));
        return false;
    };
    let args = QwenRopeRowsArgs {
        rows,
        heads,
        head_dim,
        rope_dim,
        position,
        stride,
        base_freq,
        pad: 0,
    };
    let total = rows * heads * (rope_dim / 2);
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, values, values_offset);
    enc.dispatch(((total as usize).max(1), 1, 1), (256, 1, 1));
    true
}

pub fn op_qwen_cache_write_f32(
    enc: &ComputeEncoder,
    dev: &Device,
    src: &Buffer,
    src_offset: usize,
    cache: &Buffer,
    position: i32,
    heads: i32,
    head_dim: i32,
    max_context: i32,
) -> bool {
    let name = "embed_native_qwen_cache_write_f32";
    let Some(pso) = dev.pipeline(name) else {
        set_last_error(format!(
            "op_qwen_cache_write_f32: pipeline `{name}` not found"
        ));
        return false;
    };
    let args = QwenCacheArgs {
        position,
        heads,
        head_dim,
        max_context,
    };
    let total = heads * head_dim;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, src, src_offset);
    enc.set_buffer(2, cache, 0);
    enc.dispatch(((total as usize).max(1), 1, 1), (256, 1, 1));
    true
}

#[allow(clippy::too_many_arguments)]
pub fn op_qwen_cache_write_rows_f32(
    enc: &ComputeEncoder,
    dev: &Device,
    src: &Buffer,
    src_offset: usize,
    cache: &Buffer,
    rows: i32,
    position: i32,
    heads: i32,
    head_dim: i32,
    max_context: i32,
    src_stride: i32,
) -> bool {
    let name = "embed_native_qwen_cache_write_rows_f32";
    let Some(pso) = dev.pipeline(name) else {
        set_last_error(format!(
            "op_qwen_cache_write_rows_f32: pipeline `{name}` not found"
        ));
        return false;
    };
    let args = QwenCacheRowsArgs {
        rows,
        position,
        heads,
        head_dim,
        max_context,
        src_stride,
        pad0: 0,
        pad1: 0,
    };
    let total = rows * heads * head_dim;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, src, src_offset);
    enc.set_buffer(2, cache, 0);
    enc.dispatch(((total as usize).max(1), 1, 1), (256, 1, 1));
    true
}

#[allow(clippy::too_many_arguments)]
pub fn op_qwen_attention_scores_decode_f32(
    enc: &ComputeEncoder,
    dev: &Device,
    q: &Buffer,
    k_cache: &Buffer,
    scores: &Buffer,
    position: i32,
    q_heads: i32,
    kv_heads: i32,
    head_dim: i32,
    max_context: i32,
    scale: f32,
) -> bool {
    let name = "embed_native_qwen_attention_scores_decode_f32";
    let Some(pso) = dev.pipeline(name) else {
        set_last_error(format!(
            "op_qwen_attention_scores_decode_f32: pipeline `{name}` not found"
        ));
        return false;
    };
    let args = QwenAttentionArgs {
        position,
        q_heads,
        kv_heads,
        dim: head_dim,
        max_context,
        scale,
        pad0: 0,
        pad1: 0,
    };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, q, 0);
    enc.set_buffer(2, k_cache, 0);
    enc.set_buffer(3, scores, 0);
    enc.set_threadgroup_memory_size(256 * 4, 0);
    enc.dispatch_threadgroups(((position + 1) as usize, q_heads as usize, 1), (256, 1, 1));
    true
}

#[allow(clippy::too_many_arguments)]
pub fn op_qwen_attention_scores_rows_f32(
    enc: &ComputeEncoder,
    dev: &Device,
    q: &Buffer,
    q_offset: usize,
    k_cache: &Buffer,
    scores: &Buffer,
    rows: i32,
    position: i32,
    q_heads: i32,
    kv_heads: i32,
    head_dim: i32,
    max_context: i32,
    q_stride: i32,
    score_stride: i32,
    scale: f32,
) -> bool {
    let name = "embed_native_qwen_attention_scores_rows_f32";
    let Some(pso) = dev.pipeline(name) else {
        set_last_error(format!(
            "op_qwen_attention_scores_rows_f32: pipeline `{name}` not found"
        ));
        return false;
    };
    let args = QwenAttentionRowsArgs {
        rows,
        position,
        q_heads,
        kv_heads,
        dim: head_dim,
        max_context,
        q_stride,
        score_stride,
        scale,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, q, q_offset);
    enc.set_buffer(2, k_cache, 0);
    enc.set_buffer(3, scores, 0);
    enc.set_threadgroup_memory_size(256 * 4, 0);
    enc.dispatch_threadgroups(
        ((position + rows) as usize, q_heads as usize, rows as usize),
        (256, 1, 1),
    );
    true
}

pub fn op_qwen_softmax_decode_f32(
    enc: &ComputeEncoder,
    dev: &Device,
    scores: &Buffer,
    position: i32,
    heads: i32,
    max_context: i32,
) -> bool {
    let name = "embed_native_qwen_softmax_decode_f32";
    let Some(pso) = dev.pipeline(name) else {
        set_last_error(format!(
            "op_qwen_softmax_decode_f32: pipeline `{name}` not found"
        ));
        return false;
    };
    let args = QwenAttentionArgs {
        position,
        q_heads: heads,
        kv_heads: 1,
        dim: 0,
        max_context,
        scale: 1.0,
        pad0: 0,
        pad1: 0,
    };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, scores, 0);
    enc.set_threadgroup_memory_size(256 * 4, 0);
    enc.dispatch_threadgroups((heads as usize, 1, 1), (256, 1, 1));
    true
}

#[allow(clippy::too_many_arguments)]
pub fn op_qwen_softmax_rows_f32(
    enc: &ComputeEncoder,
    dev: &Device,
    scores: &Buffer,
    rows: i32,
    position: i32,
    heads: i32,
    max_context: i32,
    score_stride: i32,
) -> bool {
    let name = "embed_native_qwen_softmax_rows_f32";
    let Some(pso) = dev.pipeline(name) else {
        set_last_error(format!(
            "op_qwen_softmax_rows_f32: pipeline `{name}` not found"
        ));
        return false;
    };
    let args = QwenAttentionRowsArgs {
        rows,
        position,
        q_heads: heads,
        kv_heads: 1,
        dim: 0,
        max_context,
        q_stride: 0,
        score_stride,
        scale: 1.0,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, scores, 0);
    enc.set_threadgroup_memory_size(256 * 4, 0);
    enc.dispatch_threadgroups((rows as usize, heads as usize, 1), (256, 1, 1));
    true
}

#[allow(clippy::too_many_arguments)]
pub fn op_qwen_attention_values_decode_f32(
    enc: &ComputeEncoder,
    dev: &Device,
    scores: &Buffer,
    v_cache: &Buffer,
    out: &Buffer,
    position: i32,
    q_heads: i32,
    kv_heads: i32,
    value_dim: i32,
    max_context: i32,
) -> bool {
    let name = "embed_native_qwen_attention_values_decode_f32";
    let Some(pso) = dev.pipeline(name) else {
        set_last_error(format!(
            "op_qwen_attention_values_decode_f32: pipeline `{name}` not found"
        ));
        return false;
    };
    let args = QwenAttentionArgs {
        position,
        q_heads,
        kv_heads,
        dim: value_dim,
        max_context,
        scale: 1.0,
        pad0: 0,
        pad1: 0,
    };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, scores, 0);
    enc.set_buffer(2, v_cache, 0);
    enc.set_buffer(3, out, 0);
    enc.set_threadgroup_memory_size(256 * 4, 0);
    enc.dispatch_threadgroups((q_heads as usize, value_dim as usize, 1), (256, 1, 1));
    true
}

#[allow(clippy::too_many_arguments)]
pub fn op_qwen_attention_values_rows_f32(
    enc: &ComputeEncoder,
    dev: &Device,
    scores: &Buffer,
    v_cache: &Buffer,
    out: &Buffer,
    rows: i32,
    position: i32,
    q_heads: i32,
    kv_heads: i32,
    value_dim: i32,
    max_context: i32,
    score_stride: i32,
) -> bool {
    let name = "embed_native_qwen_attention_values_rows_f32";
    let Some(pso) = dev.pipeline(name) else {
        set_last_error(format!(
            "op_qwen_attention_values_rows_f32: pipeline `{name}` not found"
        ));
        return false;
    };
    let args = QwenAttentionRowsArgs {
        rows,
        position,
        q_heads,
        kv_heads,
        dim: value_dim,
        max_context,
        q_stride: 0,
        score_stride,
        scale: 1.0,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, scores, 0);
    enc.set_buffer(2, v_cache, 0);
    enc.set_buffer(3, out, 0);
    enc.set_threadgroup_memory_size(256 * 4, 0);
    enc.dispatch_threadgroups(
        (rows as usize, q_heads as usize, value_dim as usize),
        (256, 1, 1),
    );
    true
}

pub fn op_qwen_apply_silu_gate_rows_f32(
    enc: &ComputeEncoder,
    dev: &Device,
    values: &Buffer,
    values_offset: usize,
    gate: &Buffer,
    gate_offset: usize,
    rows: i32,
    width: i32,
    value_stride: i32,
    gate_stride: i32,
) -> bool {
    let name = "embed_native_qwen_apply_silu_gate_rows_f32";
    let Some(pso) = dev.pipeline(name) else {
        set_last_error(format!(
            "op_qwen_apply_silu_gate_rows_f32: pipeline `{name}` not found"
        ));
        return false;
    };
    let args = QwenGateRowsArgs {
        rows,
        width,
        value_stride,
        gate_stride,
    };
    let total = rows * width;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, values, values_offset);
    enc.set_buffer(2, gate, gate_offset);
    enc.dispatch(((total as usize).max(1), 1, 1), (256, 1, 1));
    true
}

pub fn op_qwen_apply_sigmoid_gate_rows_f32(
    enc: &ComputeEncoder,
    dev: &Device,
    values: &Buffer,
    values_offset: usize,
    gate: &Buffer,
    gate_offset: usize,
    rows: i32,
    width: i32,
    value_stride: i32,
    gate_stride: i32,
) -> bool {
    let name = "embed_native_qwen_apply_sigmoid_gate_rows_f32";
    let Some(pso) = dev.pipeline(name) else {
        set_last_error(format!(
            "op_qwen_apply_sigmoid_gate_rows_f32: pipeline `{name}` not found"
        ));
        return false;
    };
    let args = QwenGateRowsArgs {
        rows,
        width,
        value_stride,
        gate_stride,
    };
    let total = rows * width;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, values, values_offset);
    enc.set_buffer(2, gate, gate_offset);
    enc.dispatch(((total as usize).max(1), 1, 1), (256, 1, 1));
    true
}

/// Byte-exact port of `ggml_metal_op_l2_norm` (ggml-metal-ops.cpp:3215-...).
#[allow(clippy::too_many_arguments)]
pub fn op_l2_norm(
    enc: &ComputeEncoder,
    dev: &Device,
    tsrc: GgmlType,
    src: &Buffer,
    dst: &Buffer,
    eps: f32,
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
) -> bool {
    let name = pipeline_name_l2_norm(tsrc);
    let Some(pso) = dev.pipeline(&name) else {
        set_last_error(format!("op_l2_norm: pipeline `{name}` not found"));
        return false;
    };
    let args = kargs::KargsL2Norm {
        ne00,
        ne01,
        ne02,
        ne03,
        nb00,
        nb01,
        nb02,
        nb03,
        ne0,
        ne1,
        ne2,
        ne3,
        nb0,
        nb1,
        nb2,
        nb3,
        eps,
    };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, src, 0);
    enc.set_buffer(2, dst, 0);
    enc.set_threadgroup_memory_size(32 * 4, 0);
    // Grid: ref ggml-metal-ops.cpp:3266 — dispatch(ne01, ne02, ne03, nth, 1, 1).
    const NTH: i32 = 512;
    enc.dispatch_threadgroups(
        (ne01 as usize, ne02 as usize, ne03 as usize),
        (ne00.min(NTH) as usize, 1, 1),
    );
    true
}

/// Byte-exact port of `ggml_metal_op_norm` (single-op, non-fused
/// path; ggml-metal-ops.cpp:3334-3470). For `n_fuse > 1` the C++
/// original chains rms_norm+mul or rms_norm+mul+add; the Rust API
/// always dispatches the plain `kernel_rms_norm_<type>[_4]` variant.
#[allow(clippy::too_many_arguments)]
pub fn op_rms_norm(
    enc: &ComputeEncoder,
    dev: &Device,
    tsrc: GgmlType,
    src: &Buffer,
    dst: &Buffer,
    eps: f32,
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
) -> bool {
    let vec4 = ne00 % 4 == 0;
    let Some(name) = pipeline_name_rms_norm(tsrc, 1, vec4) else {
        return false;
    };
    let Some(pso) = dev.pipeline(&name) else {
        set_last_error(format!("op_rms_norm: pipeline `{name}` not found"));
        return false;
    };
    // Byte-exact struct fill from ggml-metal-ops.cpp:3352-3365.
    let args = kargs::KargsNorm {
        ne00,
        ne00_t: if ne00 % 4 == 0 { ne00 / 4 } else { ne00 },
        nb1,
        nb2,
        nb3,
        eps,
        nef1: [ne01, 0, 0],
        nef2: [ne02, 0, 0],
        nef3: [ne03, 0, 0],
        nbf1: [nb01, 0, 0],
        nbf2: [nb02, 0, 0],
        nbf3: [nb03, 0, 0],
    };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, src, 0);
    enc.set_buffer(2, src, 0);
    enc.set_buffer(3, src, 0);
    enc.set_buffer(4, dst, 0);
    enc.set_threadgroup_memory_size(32 * 4, 0);
    // Grid: ref ggml-metal-ops.cpp:3450-... — dispatch(ne01, ne02, ne03, nth, 1, 1).
    const NTH: i32 = 512;
    let nth_max = if vec4 { NTH } else { 32 };
    enc.dispatch_threadgroups(
        (ne01 as usize, ne02 as usize, ne03 as usize),
        (nth_max.min(ne00) as usize, 1, 1),
    );
    true
}

/// Byte-exact port of the fused `rms_norm + mul` path in
/// `ggml_metal_op_norm` (`n_fuse = 2`). Used by Gemma RMSNorm weights.
#[allow(clippy::too_many_arguments)]
pub fn op_rms_norm_mul(
    enc: &ComputeEncoder,
    dev: &Device,
    tsrc: GgmlType,
    src: &Buffer,
    mul: &Buffer,
    mul_offset: usize,
    dst: &Buffer,
    eps: f32,
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    mul_ne01: i32,
    mul_ne02: i32,
    mul_ne03: i32,
    mul_nb01: u64,
    mul_nb02: u64,
    mul_nb03: u64,
) -> bool {
    let vec4 = ne00 % 4 == 0;
    let Some(name) = pipeline_name_rms_norm(tsrc, 2, vec4) else {
        return false;
    };
    let Some(pso) = dev.pipeline(&name) else {
        set_last_error(format!("op_rms_norm_mul: pipeline `{name}` not found"));
        return false;
    };
    let args = kargs::KargsNorm {
        ne00,
        ne00_t: if vec4 { ne00 / 4 } else { ne00 },
        nb1,
        nb2,
        nb3,
        eps,
        nef1: [ne01, mul_ne01, 0],
        nef2: [ne02, mul_ne02, 0],
        nef3: [ne03, mul_ne03, 0],
        nbf1: [nb01, mul_nb01, 0],
        nbf2: [nb02, mul_nb02, 0],
        nbf3: [nb03, mul_nb03, 0],
    };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, src, 0);
    enc.set_buffer(2, mul, mul_offset);
    enc.set_buffer(3, src, 0);
    enc.set_buffer(4, dst, 0);
    enc.set_threadgroup_memory_size(32 * 4, 0);
    const NTH: i32 = 512;
    let nth_max = if vec4 { NTH } else { 32 };
    enc.dispatch_threadgroups(
        (ne01 as usize, ne02 as usize, ne03 as usize),
        (nth_max.min(ne00).max(1) as usize, 1, 1),
    );
    true
}

/// Byte-exact port of the fused `rms_norm + mul + add` path in
/// `ggml_metal_op_norm` (`n_fuse = 3`). Used for Gemma residual joins
/// after post-attention and post-FFN norms.
#[allow(clippy::too_many_arguments)]
pub fn op_rms_norm_mul_add(
    enc: &ComputeEncoder,
    dev: &Device,
    tsrc: GgmlType,
    src: &Buffer,
    mul: &Buffer,
    mul_offset: usize,
    add: &Buffer,
    dst: &Buffer,
    eps: f32,
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    mul_ne01: i32,
    mul_ne02: i32,
    mul_ne03: i32,
    mul_nb01: u64,
    mul_nb02: u64,
    mul_nb03: u64,
    add_ne01: i32,
    add_ne02: i32,
    add_ne03: i32,
    add_nb01: u64,
    add_nb02: u64,
    add_nb03: u64,
) -> bool {
    let vec4 = ne00 % 4 == 0;
    let Some(name) = pipeline_name_rms_norm(tsrc, 3, vec4) else {
        return false;
    };
    let Some(pso) = dev.pipeline(&name) else {
        set_last_error(format!("op_rms_norm_mul_add: pipeline `{name}` not found"));
        return false;
    };
    let args = kargs::KargsNorm {
        ne00,
        ne00_t: if vec4 { ne00 / 4 } else { ne00 },
        nb1,
        nb2,
        nb3,
        eps,
        nef1: [ne01, mul_ne01, add_ne01],
        nef2: [ne02, mul_ne02, add_ne02],
        nef3: [ne03, mul_ne03, add_ne03],
        nbf1: [nb01, mul_nb01, add_nb01],
        nbf2: [nb02, mul_nb02, add_nb02],
        nbf3: [nb03, mul_nb03, add_nb03],
    };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, src, 0);
    enc.set_buffer(2, mul, mul_offset);
    enc.set_buffer(3, add, 0);
    enc.set_buffer(4, dst, 0);
    enc.set_threadgroup_memory_size(32 * 4, 0);
    const NTH: i32 = 512;
    let nth_max = if vec4 { NTH } else { 32 };
    enc.dispatch_threadgroups(
        (ne01 as usize, ne02 as usize, ne03 as usize),
        (nth_max.min(ne00).max(1) as usize, 1, 1),
    );
    true
}

/// Dispatch for the Metal `mul_mm` path.
///
/// M5/Metal4 devices use the tensor-unit metallib and its 64x128 output
/// tile. Older Apple Silicon uses the Metal3 simdgroup metallib with the
/// upstream 64x32 dispatch contract.
#[allow(clippy::too_many_arguments)]
pub fn op_mul_mm(
    enc: &ComputeEncoder,
    dev: &Device,
    tsrc0: GgmlType,
    tsrc1: GgmlType,
    src0: &Buffer,
    src0_offset: usize,
    src1: &Buffer,
    dst: &Buffer,
    // src0 dims
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    // src1 dims
    ne11: i32,
    ne12: i32,
    ne13: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    // dst dims
    ne0: i32,
    ne1: i32,
) -> bool {
    const NUM_THREADS: usize = 128;

    let r2: i16 = (ne12 / ne02) as i16;
    let r3: i16 = (ne13 / ne03) as i16;

    let tensor_path = dev.uses_tensor_mul_mm();
    let nra = 64;
    let nrb = if tensor_path { 128 } else { 32 };
    let bc_inp = ne00 % 32 != 0;
    let bc_out = !tensor_path && (ne0 % 64 != 0 || ne1 % 32 != 0);
    let threadgroup_memory_bytes = if tensor_path {
        4096
    } else if bc_out {
        8192
    } else {
        4096 + 2048
    };

    let base = format!("kernel_mul_mm_{}_{}", tsrc0.name(), tsrc1.name());
    let cache_key = format!(
        "{base}_path={}_bci={}_bco={}",
        if tensor_path { "tensor" } else { "simdgroup" },
        bc_inp as i32,
        bc_out as i32
    );

    let Some(pso) = dev.pipeline_with_constants(&cache_key, &base, |cv| {
        cv_set_bool(cv, bc_inp, FC_MUL_MM + 0);
        cv_set_bool(cv, bc_out, FC_MUL_MM + 1);
    }) else {
        set_last_error(format!("op_mul_mm: pipeline `{base}` not found"));
        return false;
    };

    let args = kargs::KargsMulMm {
        ne00,
        ne02,
        nb01,
        nb02,
        nb03,
        ne12,
        nb10,
        nb11,
        nb12,
        nb13,
        ne0,
        ne1,
        r2,
        r3,
    };

    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, src0, src0_offset);
    enc.set_buffer(2, src1, 0);
    enc.set_buffer(3, dst, 0);

    enc.set_threadgroup_memory_size(threadgroup_memory_bytes, 0);

    enc.dispatch_threadgroups(
        (
            ((ne11 + nrb - 1) / nrb) as usize,
            ((ne01 + nra - 1) / nra) as usize,
            (ne12 * ne13) as usize,
        ),
        (NUM_THREADS, 1, 1),
    );
    true
}

/// Byte-exact port of `ggml_metal_op_gated_delta_net`
/// (ggml-metal-ops.cpp:1584-1654). This is the **upstream ggml**
/// gated-delta-net op — distinct from the dflash-tape variant which
/// stays byte-exact vendored from `dflash_mlx/kernels.py` at
/// `vendor/metal/shaders/dflash/gated_delta_tape.metal`.
///
/// Pipeline picks a `_<nsg>` suffix where nsg = ne20/32, plus two
/// int16 function constants (ne20, ne30) at FC_GATED_DELTA_NET+0/1.
#[allow(clippy::too_many_arguments)]
pub fn op_gated_delta_net(
    enc: &ComputeEncoder,
    dev: &Device,
    tsrc0: GgmlType,
    src_q: &Buffer,
    src_k: &Buffer,
    src_v: &Buffer,
    src_gate: &Buffer,
    src_beta: &Buffer,
    src_state: &Buffer,
    dst: &Buffer,
    // src0 (q) dims
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    // src1 (k) dims
    ne10: i32,
    ne11: i32,
    ne12: i32,
    ne13: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    // src2 (v) dims — S_v = ne20, H = ne21
    ne20: i32,
    ne21: i32,
    ne22: i32,
    ne23: i32,
    nb20: u64,
    nb21: u64,
    nb22: u64,
    nb23: u64,
    // src3 (gate) dims — G = ne30
    ne30: i32,
    // dst dims
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
) -> bool {
    // ref: ggml-metal-device.cpp:585-617. nsg = ne20/32.
    let nsg = ne20 / 32;
    let base = format!("kernel_gated_delta_net_{}_{nsg}", tsrc0.name());
    let cache_key = format!("{base}_ne20={ne20}_ne30={ne30}");

    let Some(pso) = dev.pipeline_with_constants(&cache_key, &base, |cv| {
        cv_set_int16(cv, ne20 as i16, FC_GATED_DELTA_NET + 0);
        cv_set_int16(cv, ne30 as i16, FC_GATED_DELTA_NET + 1);
    }) else {
        set_last_error(format!("op_gated_delta_net: pipeline `{base}` not found"));
        return false;
    };

    // Byte-exact struct fill from ggml-metal-ops.cpp:1604-1641.
    // ns{02,12,22} = nb{02,12,22} / sizeof(float).
    let args = kargs::KargsGatedDeltaNet {
        ne00,
        ne01,
        ne02,
        ne03,
        nb00,
        nb01,
        nb02,
        nb03,
        ne10,
        ne11,
        ne12,
        ne13,
        nb10,
        nb11,
        nb12,
        nb13,
        ne20,
        ne21,
        ne22,
        ne23,
        nb20,
        nb21,
        nb22,
        nb23,
        ns02: (nb02 / 4) as i32,
        ns12: (nb12 / 4) as i32,
        ns22: (nb22 / 4) as i32,
        ne0,
        ne1,
        ne2,
        ne3,
        nb0,
        nb1,
        nb2,
        nb3,
    };

    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, src_q, 0);
    enc.set_buffer(2, src_k, 0);
    enc.set_buffer(3, src_v, 0);
    enc.set_buffer(4, src_gate, 0);
    enc.set_buffer(5, src_beta, 0);
    enc.set_buffer(6, src_state, 0);
    enc.set_buffer(7, dst, 0);

    // ref: ggml-metal-ops.cpp:1653 — dispatch(ne20/nsg, ne21, ne23, 32, nsg, 1).
    enc.dispatch_threadgroups(
        ((ne20 / nsg) as usize, ne21 as usize, ne23 as usize),
        (32, nsg as usize, 1),
    );
    true
}

/// Byte-exact port of `ggml_metal_op_concat` (ggml-metal-ops.cpp:516-572).
/// Concatenates two tensors along `dim` (0..3).
#[allow(clippy::too_many_arguments)]
pub fn op_concat(
    enc: &ComputeEncoder,
    dev: &Device,
    src0: &Buffer,
    src1: &Buffer,
    dst: &Buffer,
    dim: i32,
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    ne13: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
) -> bool {
    let Some(pso) = dev.pipeline("kernel_concat") else {
        set_last_error("op_concat: pipeline `kernel_concat` not found".to_string());
        return false;
    };
    let args = kargs::KargsConcat {
        ne00,
        ne01,
        ne02,
        ne03,
        nb00,
        nb01,
        nb02,
        nb03,
        ne10,
        ne11,
        ne12,
        ne13,
        nb10,
        nb11,
        nb12,
        nb13,
        ne0,
        ne1,
        ne2,
        ne3,
        nb0,
        nb1,
        nb2,
        nb3,
        dim,
    };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, src0, 0);
    enc.set_buffer(2, src1, 0);
    enc.set_buffer(3, dst, 0);
    let nth = 1024.min(ne0) as usize;
    enc.dispatch_threadgroups((ne1 as usize, ne2 as usize, ne3 as usize), (nth, 1, 1));
    true
}

/// Byte-exact port of `ggml_metal_op_repeat` (ggml-metal-ops.cpp:574-).
/// Broadcast-copies `src` into `dst` where dst dims are integer
/// multiples of src dims.
#[allow(clippy::too_many_arguments)]
pub fn op_repeat(
    enc: &ComputeEncoder,
    dev: &Device,
    tsrc: GgmlType,
    src: &Buffer,
    dst: &Buffer,
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
) -> bool {
    let name = format!("kernel_repeat_{}", tsrc.name());
    let Some(pso) = dev.pipeline(&name) else {
        set_last_error(format!("op_repeat: pipeline `{name}` not found"));
        return false;
    };
    let args = kargs::KargsRepeat {
        ne00,
        ne01,
        ne02,
        ne03,
        nb00,
        nb01,
        nb02,
        nb03,
        ne0,
        ne1,
        ne2,
        ne3,
        nb0,
        nb1,
        nb2,
        nb3,
    };
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, src, 0);
    enc.set_buffer(2, dst, 0);
    let nth = 1024.min(ne0) as usize;
    enc.dispatch_threadgroups((ne1 as usize, ne2 as usize, ne3 as usize), (nth, 1, 1));
    true
}

// ─── flash_attn_ext support ────────────────────────────────────────
//
// ref: ggml-metal-impl.h:93-97 — queries-per-threadgroup /
//      cache-values-per-simdgroup constants.

pub const OP_FLASH_ATTN_EXT_NQPSG: i32 = 8;
pub const OP_FLASH_ATTN_EXT_NCPSG: i32 = 64;
pub const OP_FLASH_ATTN_EXT_VEC_NQPSG: i32 = 1;
pub const OP_FLASH_ATTN_EXT_VEC_NCPSG: i32 = 32;

/// Byte-exact port of `ggml_metal_op_flash_attn_ext_use_vec`
/// (ggml-metal-ops.cpp:2502-2510). Picks the vec (decode) vs full
/// (prefill) path:
///
/// ```c
/// return (ne01 < 20) && (ne00 % 32 == 0);
/// ```
pub fn flash_attn_ext_use_vec(ne00: i32, ne01: i32) -> bool {
    ne01 < 20 && ne00 % 32 == 0
}

/// Byte-exact port of `ggml_metal_op_flash_attn_ext` (ggml-metal-ops.cpp:2626-3050).
///
/// # Path coverage
///
///   * **Non-vec (prefill)**: fully implemented below. Runs optional
///     pad stage (when `ne11 % ncpsg != 0`), optional mask-block
///     stage (when `has_mask`), then the main flash-attn-ext kernel.
///   * **Vec (decode)**: optional pad stage, main vec kernel, and
///     `nwg > 1` reduce pass, matching ggml's current `nwg = 32`
///     selection.
///
/// Size helpers are exposed so the caller can preallocate the three
/// scratch buffers (`pad`, `blk`, `tmp`).
#[allow(clippy::too_many_arguments)]
pub fn op_flash_attn_ext(
    enc: &ComputeEncoder,
    dev: &Device,
    tsrc1: GgmlType,        // K type (== V type per GGML_ASSERT at line 2647)
    q: &Buffer,             // src[0] — queries [dk, ne01, n_heads, batch]
    k: &Buffer,             // src[1] — keys
    v: &Buffer,             // src[2] — values
    mask: Option<&Buffer>,  // src[3] — optional additive mask (f16)
    sinks: Option<&Buffer>, // src[4] — optional attention sinks
    pad_buf: &Buffer,       // scratch (size from flash_attn_ext_extra_pad_bytes)
    blk_buf: &Buffer,       // scratch (size from flash_attn_ext_extra_blk_bytes)
    tmp_buf: &Buffer,       // scratch (size from flash_attn_ext_extra_tmp_bytes)
    dst: &Buffer,
    // q (src[0]) dims
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    // k (src[1]) dims
    ne11: i32,
    ne12: i32,
    ne13: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    // v (src[2]) dims
    ne20: i32,
    nb20: u64,
    nb21: u64,
    nb22: u64,
    nb23: u64,
    // mask (src[3]) dims — pass 0 when mask is None
    ne30: i32,
    ne31: i32,
    ne32: i32,
    ne33: i32,
    nb31: u64,
    nb32: u64,
    nb33: u64,
    // dst dims
    ne1: i32,
    ne2: i32,
    ne3: i32,
    // op params (from op_params)
    scale_in: f32,
    max_bias: f32,
    logit_softcap: f32,
) -> bool {
    // ref: ggml-metal-ops.cpp:2664-2668. Normalize scale by softcap if set.
    let scale = if logit_softcap != 0.0 {
        scale_in / logit_softcap
    } else {
        scale_in
    };
    let has_mask = mask.is_some();
    let has_sinks = sinks.is_some();
    let has_bias = max_bias != 0.0;
    let has_scap = logit_softcap != 0.0;

    // n_head_log2 = 1 << floor(log2(n_head)); m0,m1 from max_bias.
    // ref: ggml-metal-ops.cpp:2679-2682.
    let n_head = ne02.max(1) as u32;
    let n_head_log2: i32 = 1 << (31 - n_head.leading_zeros()) as i32;
    let m0 = 2.0f32.powf(-max_bias / n_head_log2 as f32);
    let m1 = 2.0f32.powf(-(max_bias / 2.0) / n_head_log2 as f32);

    // Fallback buffer when mask/sinks are absent — bind the query
    // tensor as a safe sentinel (matches the C++ fallback at
    // lines 2686-2687: `bid_src3 = has_mask ? mask : bid_src0`).
    let bid_src3 = mask.unwrap_or(q);
    let bid_src4 = sinks.unwrap_or(q);
    let ns10 = (nb11 / nb10) as i32;
    let ns20 = (nb21 / nb20) as i32;

    // ─── Vec (decode) path ───
    if tsrc1 != GgmlType::F16 && flash_attn_ext_use_vec(ne00, ne01) {
        let nqptg = OP_FLASH_ATTN_EXT_VEC_NQPSG;
        let ncpsg = OP_FLASH_ATTN_EXT_VEC_NCPSG;
        let nhptg = 1;
        let has_kvpad = ne11 % ncpsg != 0;

        if has_kvpad {
            let args0 = kargs::KargsFlashAttnExtPad {
                ne11,
                ne_12_2: ne12,
                ne_12_3: ne13,
                nb11,
                nb12,
                nb13,
                nb21,
                nb22,
                nb23,
                ne31,
                ne32,
                ne33,
                nb31,
                nb32,
                nb33,
            };
            let pad_base = "kernel_flash_attn_ext_pad";
            let pad_key = format!("{pad_base}_mask={}_ncpsg={ncpsg}", has_mask as i32);
            let Some(pso) = dev.pipeline_with_constants(&pad_key, pad_base, |cv| {
                cv_set_bool(cv, has_mask, FC_FLASH_ATTN_EXT_PAD + 0);
                cv_set_int32(cv, ncpsg, FC_FLASH_ATTN_EXT_PAD + 25);
            }) else {
                set_last_error(format!(
                    "op_flash_attn_ext: vec pad pipeline `{pad_base}` not found"
                ));
                return false;
            };
            enc.set_pipeline(&pso);
            enc.set_bytes(0, &args0);
            enc.set_buffer(1, k, 0);
            enc.set_buffer(2, v, 0);
            enc.set_buffer(3, bid_src3, 0);
            enc.set_buffer(4, pad_buf, 0);
            enc.dispatch_threadgroups(
                (
                    ncpsg as usize,
                    ne12.max(ne32) as usize,
                    ne13.max(ne33) as usize,
                ),
                (32, 1, 1),
            );
        }

        if has_kvpad {
            enc.memory_barrier_buffers();
        }

        if ne00 < ne20 {
            set_last_error(format!(
                "op_flash_attn_ext: vec path requires K head dim >= V head dim, got dk={ne00} dv={ne20}"
            ));
            return false;
        }

        let mut nsg = 1;
        let nwg = 32;
        while 2 * nwg * nsg * ncpsg < ne11 && nsg < 4 {
            nsg *= 2;
        }

        let args = kargs::KargsFlashAttnExtVec {
            ne01,
            ne02,
            ne03,
            nb01,
            nb02,
            nb03,
            ne11,
            ne_12_2: ne12,
            ne_12_3: ne13,
            ns10,
            nb11,
            nb12,
            nb13,
            ns20,
            nb21,
            nb22,
            nb23,
            ne31,
            ne32,
            ne33,
            nb31,
            nb32,
            nb33,
            ne1,
            ne2,
            ne3,
            scale,
            max_bias,
            m0,
            m1,
            n_head_log2,
            logit_softcap,
        };

        let dk = ne00;
        let dv = ne20;
        let vec_base = format!("kernel_flash_attn_ext_vec_{}_dk{dk}_dv{dv}", tsrc1.name());
        let vec_key = format!(
            "{vec_base}_mask={}_sinks={}_bias={}_scap={}_kvpad={}_ns10={ns10}_ns20={ns20}_nsg={nsg}_nwg={nwg}",
            has_mask as i32, has_sinks as i32, has_bias as i32,
            has_scap as i32, has_kvpad as i32,
        );
        let Some(pso) = dev.pipeline_with_constants(&vec_key, &vec_base, |cv| {
            cv_set_bool(cv, has_mask, FC_FLASH_ATTN_EXT_VEC + 0);
            cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_VEC + 1);
            cv_set_bool(cv, has_bias, FC_FLASH_ATTN_EXT_VEC + 2);
            cv_set_bool(cv, has_scap, FC_FLASH_ATTN_EXT_VEC + 3);
            cv_set_bool(cv, has_kvpad, FC_FLASH_ATTN_EXT_VEC + 4);
            cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_VEC + 20);
            cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_VEC + 21);
            cv_set_int32(cv, nsg, FC_FLASH_ATTN_EXT_VEC + 22);
            cv_set_int32(cv, nwg, FC_FLASH_ATTN_EXT_VEC + 23);
        }) else {
            set_last_error(format!(
                "op_flash_attn_ext: vec pipeline `{vec_base}` not found"
            ));
            return false;
        };

        let pad128_dk = ((ne00 + 127) / 128) * 128;
        let pad128_dv = ((ne20 + 127) / 128) * 128;
        let smem_raw = ((pad128_dk + 4 * ncpsg + 2 * pad128_dv) * nsg) * 2;
        let smem = ((smem_raw + 15) / 16 * 16) as usize;

        enc.set_pipeline(&pso);
        enc.set_bytes(0, &args);
        enc.set_buffer(1, q, 0);
        enc.set_buffer(2, k, 0);
        enc.set_buffer(3, v, 0);
        enc.set_buffer(4, bid_src3, 0);
        enc.set_buffer(5, bid_src4, 0);
        enc.set_buffer(6, pad_buf, 0);
        enc.set_buffer(7, tmp_buf, 0);
        enc.set_threadgroup_memory_size(smem, 0);
        enc.dispatch_threadgroups(
            (
                ((ne01 + nqptg - 1) / nqptg) as usize,
                ((ne02 + nhptg - 1) / nhptg) as usize,
                (ne03 * nwg) as usize,
            ),
            (32, nsg as usize, 1),
        );
        enc.memory_barrier_buffers();

        let nrows = ne1 * ne2 * ne3;
        let args0 = kargs::KargsFlashAttnExtVecReduce { nrows };
        let reduce_base = "kernel_flash_attn_ext_vec_reduce";
        let reduce_key = format!("{reduce_base}_dv={dv}_nwg={nwg}");
        let Some(pso) = dev.pipeline_with_constants(&reduce_key, reduce_base, |cv| {
            cv_set_int32(cv, dv, FC_FLASH_ATTN_EXT_VEC_REDUCE + 0);
            cv_set_int32(cv, nwg, FC_FLASH_ATTN_EXT_VEC_REDUCE + 1);
        }) else {
            set_last_error(format!(
                "op_flash_attn_ext: vec reduce pipeline `{reduce_base}` not found"
            ));
            return false;
        };
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &args0);
        enc.set_buffer(1, tmp_buf, 0);
        enc.set_buffer(2, dst, 0);
        enc.dispatch_threadgroups((nrows as usize, 1, 1), ((32 * nwg) as usize, 1, 1));
        return true;
    }

    // ─── Non-vec (prefill) path ───
    let nqptg = OP_FLASH_ATTN_EXT_NQPSG;
    let ncpsg = OP_FLASH_ATTN_EXT_NCPSG;
    let has_kvpad = ne11 % ncpsg != 0;

    // Stage 1 (optional): kvpad. ref: ggml-metal-ops.cpp:2715-2753.
    if has_kvpad {
        let args0 = kargs::KargsFlashAttnExtPad {
            ne11,
            ne_12_2: ne12,
            ne_12_3: ne13,
            nb11,
            nb12,
            nb13,
            nb21,
            nb22,
            nb23,
            ne31,
            ne32,
            ne33,
            nb31,
            nb32,
            nb33,
        };
        let pad_base = "kernel_flash_attn_ext_pad";
        let pad_key = format!("{pad_base}_mask={}_ncpsg={ncpsg}", has_mask as i32);
        let Some(pso) = dev.pipeline_with_constants(&pad_key, pad_base, |cv| {
            cv_set_bool(cv, has_mask, FC_FLASH_ATTN_EXT_PAD + 0);
            cv_set_int32(cv, ncpsg, FC_FLASH_ATTN_EXT_PAD + 25);
        }) else {
            set_last_error(format!(
                "op_flash_attn_ext: pad pipeline `{pad_base}` not found"
            ));
            return false;
        };
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &args0);
        enc.set_buffer(1, k, 0);
        enc.set_buffer(2, v, 0);
        enc.set_buffer(3, bid_src3, 0);
        enc.set_buffer(4, pad_buf, 0);
        // Grid: dispatch(ncpsg, max(ne12, ne32), max(ne13, ne33), 32, 1, 1).
        // ref: ggml-metal-ops.cpp:2746.
        enc.dispatch_threadgroups(
            (
                ncpsg as usize,
                ne12.max(ne32) as usize,
                ne13.max(ne33) as usize,
            ),
            (32, 1, 1),
        );
    }

    // Stage 2 (optional): mask-block. ref: ggml-metal-ops.cpp:2755-2778.
    if has_mask {
        let args0 = kargs::KargsFlashAttnExtBlk {
            ne01,
            ne30,
            ne31,
            ne32,
            ne33,
            nb31,
            nb32,
            nb33,
        };
        let blk_base = "kernel_flash_attn_ext_blk";
        let blk_key = format!("{blk_base}_nqptg={nqptg}_ncpsg={ncpsg}");
        let Some(pso) = dev.pipeline_with_constants(&blk_key, blk_base, |cv| {
            cv_set_int32(cv, nqptg, FC_FLASH_ATTN_EXT_BLK + 24);
            cv_set_int32(cv, ncpsg, FC_FLASH_ATTN_EXT_BLK + 25);
        }) else {
            set_last_error(format!(
                "op_flash_attn_ext: blk pipeline `{blk_base}` not found"
            ));
            return false;
        };
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &args0);
        enc.set_buffer(1, bid_src3, 0);
        enc.set_buffer(2, blk_buf, 0);
        let nblk1 = (ne01 + nqptg - 1) / nqptg;
        let nblk0 = (ne30 + ncpsg - 1) / ncpsg;
        enc.dispatch_threadgroups(
            (nblk0 as usize, nblk1 as usize, (ne32 * ne33) as usize),
            (32, 1, 1),
        );
    }

    if has_kvpad || has_mask {
        enc.memory_barrier_buffers();
    }

    // Stage 3 (main): flash_attn_ext. ref: ggml-metal-ops.cpp:2802-2860.
    let is_q = matches!(tsrc1, GgmlType::Q4_K | GgmlType::Q6_K) as i32;
    // FATTN_SMEM macro — ref: ggml-metal-ops.cpp:2794.
    let pad64 = ((ne20 + 63) / 64) * 64;
    // nsg: ref ggml-metal-ops.cpp:2810.
    let nsg: i32 = if ne00 >= 512 { 8 } else { 4 };
    let inner = ne00 + 2 * pad64 + 2 * (2 * ncpsg);
    let smem_raw = (nqptg * inner + is_q * (16 * 32 * nsg)) * 2; // sizeof(float)/2 = 2
    let smem = ((smem_raw + 15) / 16 * 16) as usize;

    let bc_mask = has_mask && (ne31 % 8 != 0);

    let args = kargs::KargsFlashAttnExt {
        ne01,
        ne02,
        ne03,
        nb01,
        nb02,
        nb03,
        ne11,
        ne_12_2: ne12,
        ne_12_3: ne13,
        ns10,
        nb11,
        nb12,
        nb13,
        ns20,
        nb21,
        nb22,
        nb23,
        ne31,
        ne32,
        ne33,
        nb31,
        nb32,
        nb33,
        ne1,
        ne2,
        ne3,
        scale,
        max_bias,
        m0,
        m1,
        n_head_log2,
        logit_softcap,
    };

    let dk = ne00;
    let dv = ne20;
    let main_base = format!("kernel_flash_attn_ext_{}_dk{dk}_dv{dv}", tsrc1.name());
    let main_key = format!(
        "{main_base}_mask={}_sinks={}_bias={}_scap={}_kvpad={}_bcm={}_ns10={ns10}_ns20={ns20}_nsg={nsg}",
        has_mask as i32, has_sinks as i32, has_bias as i32,
        has_scap as i32, has_kvpad as i32, bc_mask as i32,
    );
    let Some(pso) = dev.pipeline_with_constants(&main_key, &main_base, |cv| {
        cv_set_bool(cv, has_mask, FC_FLASH_ATTN_EXT + 0);
        cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT + 1);
        cv_set_bool(cv, has_bias, FC_FLASH_ATTN_EXT + 2);
        cv_set_bool(cv, has_scap, FC_FLASH_ATTN_EXT + 3);
        cv_set_bool(cv, has_kvpad, FC_FLASH_ATTN_EXT + 4);
        cv_set_bool(cv, bc_mask, FC_FLASH_ATTN_EXT + 10);
        cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT + 20);
        cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT + 21);
        cv_set_int32(cv, nsg, FC_FLASH_ATTN_EXT + 22);
    }) else {
        set_last_error(format!(
            "op_flash_attn_ext: main pipeline `{main_base}` not found"
        ));
        return false;
    };

    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, q, 0);
    enc.set_buffer(2, k, 0);
    enc.set_buffer(3, v, 0);
    enc.set_buffer(4, bid_src3, 0);
    enc.set_buffer(5, bid_src4, 0);
    enc.set_buffer(6, pad_buf, 0);
    enc.set_buffer(7, blk_buf, 0);
    enc.set_buffer(8, dst, 0);
    enc.set_threadgroup_memory_size(smem, 0);

    // Grid: dispatch((ne01+nqptg-1)/nqptg, ne02, ne03, 32, nsg, 1).
    // ref: ggml-metal-ops.cpp:2858.
    enc.dispatch_threadgroups(
        (
            ((ne01 + nqptg - 1) / nqptg) as usize,
            ne02 as usize,
            ne03 as usize,
        ),
        (32, nsg as usize, 1),
    );

    true
}

/// Byte-exact port of `ggml_metal_op_mul_mat_id`
/// (ggml-metal-ops.cpp:2268-2420) — MoE expert-routed matmul used by
/// the 35B-A3B target. Two stages:
///
///   1. `kernel_mul_mm_id_map0_ne20_<ne20>_ne02=<ne02>` builds an
///      expert-to-token mapping table (token-per-expert counts + flat
///      token-id list) from the router `ids` tensor.
///   2. `kernel_mul_mm_id_<t0>_<t1>_bci=<bc_inp>` does the actual
///      matmul for each active expert, reading the map to know which
///      tokens to route where.
///
/// Callers supply two scratch buffers — `tpe` and `ids_map` — sized
/// by the formulas in `ggml_metal_op_mul_mat_id_extra_tpe` /
/// `_extra_ids` from the C++ source. This Rust port exposes the
/// shape-compute helpers so the caller can pre-allocate correctly.
pub fn mul_mat_id_extra_tpe_bytes(ne02: i32) -> usize {
    // ref: ggml-metal-ops.cpp:2251 — `ggml_metal_op_mul_mat_id_extra_tpe`
    //      returns `(ne02 + 1) * sizeof(int32)`.
    ((ne02 + 1) as usize) * std::mem::size_of::<i32>()
}

pub fn mul_mat_id_extra_ids_bytes(ne02: i32, ne20: i32, ne21: i32) -> usize {
    // ref: ggml-metal-ops.cpp:2259 — `ggml_metal_op_mul_mat_id_extra_ids`
    //      returns `ne02 * ne20 * ne21 * sizeof(int32)`.
    (ne02 as usize) * (ne20 as usize) * (ne21 as usize) * std::mem::size_of::<i32>()
}

#[allow(clippy::too_many_arguments)]
pub fn op_mul_mat_id(
    enc: &ComputeEncoder,
    dev: &Device,
    tsrc0: GgmlType,  // expert weight type (e.g. Q4_K)
    tsrc1: GgmlType,  // token activation type (usually F32)
    src0: &Buffer,    // [ne00, ne01, ne02] expert weights (ne02 = num_experts)
    src1: &Buffer,    // [ne10, ne11, ne12] token activations
    ids: &Buffer,     // [ne20, ne21] i32 expert-selection indices
    tpe: &Buffer,     // scratch: token-per-expert counts — size from mul_mat_id_extra_tpe_bytes
    ids_map: &Buffer, // scratch: flat token-id map       — size from mul_mat_id_extra_ids_bytes
    dst: &Buffer,
    // src0 dims (expert weights)
    ne00: i32,
    ne01: i32,
    ne02: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    // src1 dims (token activations)
    ne10: i32,
    ne11: i32,
    ne12: i32,
    ne13: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    // ids dims
    ne20: i32,
    ne21: i32,
    nb11_ids: u64,
    nb12_ids: u64,
    nb21_ids: u64,
    // dst dims
    ne0: i32,
    ne1: i32,
) -> bool {
    // ───────────────────────────────────────────────────────────
    // Stage 1: kernel_mul_mm_id_map0 — build token-per-expert map
    // ref: ggml-metal-ops.cpp:2316-2355
    // ───────────────────────────────────────────────────────────
    let map0_base = format!("kernel_mul_mm_id_map0_ne20_{ne20}");
    let map0_key = format!("{map0_base}_ne02={ne02}");
    let Some(map0_pso) = dev.pipeline(&map0_base) else {
        // Fallback: try keyed name for cache stability across calls
        let Some(_retry) = dev.pipeline(&map0_key) else {
            set_last_error(format!(
                "op_mul_mat_id: pipeline `{map0_base}` not found \
                 (kernel is instantiated per (ne02, ne20) via FC)"
            ));
            return false;
        };
        return false;
    };

    let args_map0 = kargs::KargsMulMmIdMap0 {
        ne02,
        ne10,
        ne11: ne11,
        nb11: nb11_ids,
        nb12: nb12_ids,
        ne21,
        ne20,
        nb21: nb21_ids,
    };

    // smem = ne02 * ne20 * sizeof(uint16_t). ref: ggml-metal-device.cpp:895
    let map0_smem = (ne02 as usize) * (ne20 as usize) * std::mem::size_of::<u16>();

    enc.set_pipeline(&map0_pso);
    enc.set_bytes(0, &args_map0);
    enc.set_buffer(1, ids, 0);
    enc.set_buffer(2, tpe, 0);
    enc.set_buffer(3, ids_map, 0);
    enc.set_threadgroup_memory_size(map0_smem, 0);
    // Grid: dispatch(1, 1, 1, ne02, 1, 1). ref: ggml-metal-ops.cpp:2352
    enc.dispatch_threadgroups((1, 1, 1), (ne02 as usize, 1, 1));

    // ───────────────────────────────────────────────────────────
    // Stage 2: kernel_mul_mm_id — the actual per-expert matmul
    // ref: ggml-metal-ops.cpp:2360-2398
    // ───────────────────────────────────────────────────────────
    let bc_inp = ne00 % 32 != 0;
    let mul_mm_id_base = format!("kernel_mul_mm_id_{}_{}", tsrc0.name(), tsrc1.name());
    let mul_mm_id_key = format!("{mul_mm_id_base}_bci={}", bc_inp as i32);

    let Some(mul_mm_id_pso) = dev.pipeline_with_constants(&mul_mm_id_key, &mul_mm_id_base, |cv| {
        cv_set_bool(cv, bc_inp, FC_MUL_MM + 0)
    }) else {
        set_last_error(format!(
            "op_mul_mat_id: pipeline `{mul_mm_id_base}` (bci={}) not found",
            bc_inp as i32
        ));
        return false;
    };

    let args_mm_id = kargs::KargsMulMmId {
        ne00,
        ne02,
        nb01,
        nb02,
        nb03,
        ne11, // n_expert_used (bcast)
        nb10,
        nb11,
        nb12,
        nb13,
        ne20, // n_expert_used
        ne21, // n_tokens
        ne0,
        ne1,
        r2: 1,
        r3: 1,
    };

    // smem = 8192 for the mul_mm_id kernel. ref: ggml-metal-device.cpp:923
    const MUL_MM_ID_SMEM: usize = 8192;

    enc.set_pipeline(&mul_mm_id_pso);
    enc.set_bytes(0, &args_mm_id);
    enc.set_buffer(1, src0, 0);
    enc.set_buffer(2, src1, 0);
    enc.set_buffer(3, tpe, 0);
    enc.set_buffer(4, ids_map, 0);
    enc.set_buffer(5, dst, 0);
    enc.set_threadgroup_memory_size(MUL_MM_ID_SMEM, 0);

    // Grid: dispatch((ne21+31)/32, (ne01+63)/64, ne02, 128, 1, 1).
    // ref: ggml-metal-ops.cpp:2397
    enc.dispatch_threadgroups(
        (
            ((ne21 + 31) / 32) as usize,
            ((ne01 + 63) / 64) as usize,
            ne02 as usize,
        ),
        (128, 1, 1),
    );

    // Silence borrow-checker for unused ne12/ne13 in args (they're
    // already pulled into args_mm_id above via nb12/nb13; keep the
    // params on the signature since ggml_metal_op_mul_mat_id reads
    // them even though they don't appear in the kargs struct).
    let _ = (ne12, ne13);

    true
}

/// Byte-exact port of the `mul_mv` dispatch path of
/// `ggml_metal_op_mul_mat` (ggml-metal-ops.cpp:2200-2250) — the
/// matrix-vector kernel that handles decode (ne11 = 1 small batch).
/// For prefill (ne11 > 8) the op goes through `op_mul_mm` which is
/// a separate port item.
#[allow(clippy::too_many_arguments)]
pub fn op_mul_mv(
    enc: &ComputeEncoder,
    dev: &Device,
    tsrc0: GgmlType, // weight matrix type (e.g. Q4_K)
    tsrc1: GgmlType, // activation type (usually F32)
    src0: &Buffer,   // [ne00, ne01, ne02, ne03]
    src0_offset: usize,
    src1: &Buffer, // [ne10, ne11, ne12, ne13]
    dst: &Buffer,  // [ne0, ne1, ne12, ne13]
    // src0 dims
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    // src1 dims
    ne10: i32,
    ne11: i32,
    ne12: i32,
    ne13: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    // dst dims
    ne0: i32,
    ne1: i32,
) -> bool {
    // ref: ggml-metal-ops.cpp:2047-2048
    let r2: i16 = (ne12 / ne02) as i16;
    let r3: i16 = (ne13 / ne03) as i16;

    let (name, shape) = pipeline_name_mul_mv(tsrc0, tsrc1, ne00);
    let cache_key = format!("{name}_nsg={}", shape.nsg);
    let Some(pso) = dev.pipeline_with_constants(&cache_key, &name, |cv| {
        cv_set_int16(cv, shape.nsg as i16, FC_MUL_MV + 0);
    }) else {
        set_last_error(format!("op_mul_mv: pipeline `{name}` not found"));
        return false;
    };

    let args = kargs::KargsMulMv {
        ne00,
        ne01,
        ne02,
        nb00,
        nb01,
        nb02,
        nb03,
        ne10,
        ne11,
        ne12,
        nb10,
        nb11,
        nb12,
        nb13,
        ne0,
        ne1,
        nr0: shape.nr0,
        r2,
        r3,
    };

    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, src0, src0_offset);
    enc.set_buffer(2, src1, 0);
    enc.set_buffer(3, dst, 0);
    if shape.smem > 0 {
        enc.set_threadgroup_memory_size(shape.smem, 0);
    }

    // Grid: ref ggml-metal-ops.cpp:2240-2246. Dense types (F32/F16/BF16/Q8_0)
    // dispatch with (ne01+nr0-1)/nr0; quantized types divide by nr0*nsg.
    let nsg = shape.nsg;
    let nr0 = shape.nr0;
    let nr1 = shape.nr1;
    let dense = matches!(tsrc0, GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16);
    let grid_x = if dense {
        (ne01 + nr0 - 1) / nr0
    } else {
        (ne01 + nr0 * nsg - 1) / (nr0 * nsg)
    };
    let grid_y = (ne11 + nr1 - 1) / nr1;
    let grid_z = ne12 * ne13;
    enc.dispatch_threadgroups(
        (grid_x as usize, grid_y as usize, grid_z as usize),
        (32, nsg as usize, 1),
    );
    true
}

/// Byte-exact port of `ggml_metal_op_rope`
/// (ggml-metal-ops.cpp:3472-3573). Handles M-RoPE / NeoX / Norm /
/// Vision variants via the `pipeline_name_rope_*` selector passed in.
#[allow(clippy::too_many_arguments)]
pub fn op_rope(
    enc: &ComputeEncoder,
    dev: &Device,
    pipeline_name: &str,   // e.g. "kernel_rope_multi_f32"
    src0: &Buffer,         // activations [ne00..ne03] = [head_dim, n_tok, n_heads, n_seq]
    src1: &Buffer,         // positions [ne10] i32
    src2: Option<&Buffer>, // optional freq table
    dst: &Buffer,
    // src0 shape
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    // dst shape
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    // op_params (from `ggml_rope`'s op_params)
    n_past: i32,
    n_dims: i32,
    n_ctx_orig: i32,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    sect_0: i32,
    sect_1: i32,
    sect_2: i32,
    sect_3: i32,
) -> bool {
    let Some(pso) = dev.pipeline(pipeline_name) else {
        set_last_error(format!("op_rope: pipeline `{pipeline_name}` not found"));
        return false;
    };
    // Byte-exact struct fill from ggml-metal-ops.cpp:3518-3549.
    let args = kargs::KargsRope {
        ne00,
        ne01,
        ne02,
        ne03,
        nb00,
        nb01,
        nb02,
        nb03,
        ne0,
        ne1,
        ne2,
        ne3,
        nb0,
        nb1,
        nb2,
        nb3,
        n_past,
        n_dims,
        n_ctx_orig,
        freq_base,
        freq_scale,
        ext_factor,
        attn_factor,
        beta_fast,
        beta_slow,
        sect_0,
        sect_1,
        sect_2,
        sect_3,
        src2: src2.is_some(),
    };

    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, src0, 0);
    enc.set_buffer(2, src1, 0);
    enc.set_buffer(3, src2.unwrap_or(src0), 0);
    enc.set_buffer(4, dst, 0);

    // Grid: ref ggml-metal-ops.cpp:3563 — dispatch(ne01, ne02, ne03, nth, 1, 1).
    let nth = ne00.min(1024);
    enc.dispatch_threadgroups(
        (ne01 as usize, ne02 as usize, ne03 as usize),
        (nth as usize, 1, 1),
    );
    true
}

/// Byte-exact port of `ggml_metal_op_ssm_conv`
/// (ggml-metal-ops.cpp:1378-1442). Picks the batched or unbatched
/// variant based on `ne1` (prefill length): >1 → batched with
/// power-of-2 batch size.
#[allow(clippy::too_many_arguments)]
pub fn op_ssm_conv(
    enc: &ComputeEncoder,
    dev: &Device,
    tsrc0: GgmlType, // conv_state type (f32)
    tsrc1: GgmlType, // x_new type (f32)
    src0: &Buffer,
    src1: &Buffer,
    dst: &Buffer,
    ne00: i64,
    ne01: i64,
    ne02: i64,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    ne10: i64,
    ne11: i64,
    nb10: u64,
    nb11: u64,
    ne0: i64,
    ne1: i64,
    ne2: i64,
    nb0: u64,
    nb1: u64,
    nb2: u64,
) -> bool {
    let args = kargs::KargsSsmConv {
        ne00,
        ne01,
        ne02,
        nb00,
        nb01,
        nb02,
        ne10,
        ne11,
        nb10,
        nb11,
        ne0,
        ne1,
        ne2,
        nb0,
        nb1,
        nb2,
    };

    // ref: ggml-metal-ops.cpp:1405-1442 — picks batched vs unbatched
    let use_batched = ne1 > 1;
    let suffix = if ne10 % 4 == 0 { "_4" } else { "" };

    if use_batched {
        let batch_size: i64 = if ne1 > 128 {
            256
        } else if ne1 > 64 {
            128
        } else if ne1 > 32 {
            64
        } else if ne1 > 16 {
            32
        } else if ne1 > 8 {
            16
        } else if ne1 > 4 {
            8
        } else {
            2
        };
        // ref: ggml-metal-device.cpp:490-508
        let base = format!(
            "kernel_ssm_conv_{}_{}_batched{suffix}",
            tsrc0.name(),
            tsrc1.name()
        );
        let name = format!("{base}_ssm_conv_bs={batch_size}");
        let Some(pso) = dev.pipeline(&name) else {
            set_last_error(format!("op_ssm_conv: pipeline `{name}` not found"));
            return false;
        };
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &args);
        enc.set_buffer(1, src0, 0);
        enc.set_buffer(2, src1, 0);
        enc.set_buffer(3, dst, 0);
        let n_batches = (ne1 + batch_size - 1) / batch_size;
        enc.dispatch_threadgroups(
            (ne01 as usize, n_batches as usize, ne02 as usize),
            (batch_size as usize, 1, 1),
        );
    } else {
        // ref: ggml-metal-device.cpp:462-488
        let name = format!("kernel_ssm_conv_{}_{}{suffix}", tsrc0.name(), tsrc1.name());
        let Some(pso) = dev.pipeline(&name) else {
            set_last_error(format!("op_ssm_conv: pipeline `{name}` not found"));
            return false;
        };
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &args);
        enc.set_buffer(1, src0, 0);
        enc.set_buffer(2, src1, 0);
        enc.set_buffer(3, dst, 0);
        enc.dispatch_threadgroups((ne01 as usize, ne1 as usize, ne02 as usize), (1, 1, 1));
    }
    true
}

/// Byte-exact port of `ggml_metal_op_soft_max` (ggml-metal-ops.cpp:1282-1370).
/// Dispatches `kernel_soft_max_<type>[_4]`.
#[allow(clippy::too_many_arguments)]
pub fn op_soft_max(
    enc: &ComputeEncoder,
    dev: &Device,
    tsrc: GgmlType,
    src0: &Buffer,
    src1: Option<&Buffer>, // mask (optional; default = src0 per C++ fallback)
    src2: Option<&Buffer>, // sinks (optional; default = src0)
    dst: &Buffer,
    scale: f32,
    max_bias: f32,
    // src0 shape (f32 logits, [ne00, ne01, ne02, ne03])
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    // optional mask shape (src1)
    ne11: i32,
    ne12: i32,
    ne13: i32,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    // dst strides
    nb1: u64,
    nb2: u64,
    nb3: u64,
) -> bool {
    let vec4 = ne00 % 4 == 0;
    let name = pipeline_name_soft_max(tsrc, vec4);
    let Some(pso) = dev.pipeline(&name) else {
        set_last_error(format!("op_soft_max: pipeline `{name}` not found"));
        return false;
    };

    // n_head = src[0]->ne[2]; n_head_log2 = 1 << floor(log2(n_head)).
    // ref: ggml-metal-ops.cpp:1305-1306
    let n_head = ne02.max(1) as u32;
    let n_head_log2: i32 = 1 << (31 - n_head.leading_zeros()) as i32;
    let m0 = 2.0f32.powf(-max_bias / n_head_log2 as f32);
    let m1 = 2.0f32.powf(-(max_bias / 2.0) / n_head_log2 as f32);

    let args = kargs::KargsSoftMax {
        ne00,
        ne01,
        ne02,
        nb01,
        nb02,
        nb03,
        ne11,
        nb11: nb11,
        nb12,
        nb13,
        nb1,
        nb2,
        nb3,
        scale,
        max_bias,
        m0,
        m1,
        n_head_log2,
    };
    let _ = (ne12, ne13);

    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, src0, 0);
    enc.set_buffer(2, src1.unwrap_or(src0), 0);
    enc.set_buffer(3, src2.unwrap_or(src0), 0);
    enc.set_buffer(4, dst, 0);

    // Threadgroup memory: ref ggml-metal-device.cpp soft_max helper;
    // smem = 32 * sizeof(float) for the partial-max reduction.
    enc.set_threadgroup_memory_size(32 * 4, 0);

    // Grid: ref ggml-metal-ops.cpp:1336-1346 — nth escalation, then
    // dispatch(ne01, ne02, ne03, nth, 1, 1).
    let mut nth: i32 = 32;
    let bound = if vec4 { ne00 / 4 } else { ne00 };
    while nth < bound && (nth * ne01 * ne02 * ne03) < 256 {
        nth *= 2;
    }
    enc.dispatch_threadgroups(
        (ne01 as usize, ne02 as usize, ne03 as usize),
        (nth as usize, 1, 1),
    );
    true
}

/// Byte-exact port of `ggml_metal_op_get_rows`
/// (ggml-metal-ops.cpp:1133-1175).
/// Dispatches `kernel_get_rows_<type>` — used for token embedding
/// lookup when the embed table is quantized.
#[allow(clippy::too_many_arguments)]
pub fn op_get_rows(
    enc: &ComputeEncoder,
    dev: &Device,
    tsrc: GgmlType,
    table: &Buffer, // src[0] — embed table (e.g. Q4_K weight matrix)
    table_offset: usize,
    ids: &Buffer, // src[1] — int32 token ids
    out: &Buffer, // dst
    ne00: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
) -> bool {
    let name = pipeline_name_get_rows(tsrc);
    let Some(pso) = dev.pipeline(&name) else {
        set_last_error(format!("op_get_rows: pipeline `{name}` not found"));
        return false;
    };

    // ne00t: for quantized types ne00/16 (16 = k-quant block size);
    // for dense types ne00.
    // ref: ggml-metal-ops.cpp:1148 — `ggml_is_quantized(...) ? ne00/16 : ne00`
    let is_quant = matches!(
        tsrc,
        GgmlType::Q5_0 | GgmlType::Q8_0 | GgmlType::Q4_K | GgmlType::Q6_K
    );
    let ne00t = if is_quant { ne00 / 16 } else { ne00 };

    let args = kargs::KargsGetRows {
        ne00t,
        ne00,
        nb01,
        nb02,
        nb03,
        ne10,
        nb10,
        nb11,
        nb12,
        nb1,
        nb2,
        nb3,
    };

    // nth = min(ne00t, nth_max). nth_max fallback = 1024.
    const NTH_MAX: i32 = 1024;
    let nth = ne00t.min(NTH_MAX);
    let nw0 = (ne00t + nth - 1) / nth;

    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, table, table_offset);
    enc.set_buffer(2, ids, 0);
    enc.set_buffer(3, out, 0);

    // ref: ggml-metal-ops.cpp:1172 — dispatch(nw0*ne10, ne11, ne12, nth, 1, 1).
    enc.dispatch_threadgroups(
        ((nw0 * ne10) as usize, ne11 as usize, ne12 as usize),
        (nth as usize, 1, 1),
    );
    true
}

/// Byte-exact port of `ggml_metal_op_argmax`
/// (ggml-metal-ops.cpp:4163-4199).
///
/// Dispatches `kernel_argmax_<type>` with `KargsArgmax{ne00, nb01}`,
/// threadgroup memory 32*(sizeof(float)+sizeof(i32)) = 256 bytes,
/// grid `(nrows, 1, 1)` threadgroups × `(nth, 1, 1)` threads where
/// nth is picked via the same escalation rule as the C++ original.
#[allow(clippy::too_many_arguments)]
pub fn op_argmax(
    enc: &ComputeEncoder,
    dev: &Device,
    src: &Buffer,
    dst: &Buffer,
    tsrc: GgmlType,
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne03: i64,
    nb01: u64,
) -> bool {
    let name = pipeline_name_argmax(tsrc);
    let Some(pso) = dev.pipeline(&name) else {
        set_last_error(format!(
            "op_argmax: pipeline `{name}` not found in metallib"
        ));
        return false;
    };

    // Byte-exact struct fill from ggml-metal-ops.cpp:4174-4177.
    let args = kargs::KargsArgmax { ne00, nb01 };

    // Threadgroup memory: 32 * (sizeof(float) + sizeof(int32)).
    // ref: ggml-metal-device.cpp:1117 — `res.smem = 32*(sizeof(float) + sizeof(int32_t))`
    const SMEM: usize = 32 * (4 /* sizeof float */ + 4/* sizeof i32 */);

    // nth escalation: start at 32, double while nth*nrows_cube < 256
    // and nth < ne00. ref: ggml-metal-ops.cpp:4182-4185.
    let mut nth: i64 = 32;
    while nth < ne00 && nth * ne01 * ne02 * ne03 < 256 {
        nth *= 2;
    }

    // Encoder: set pipeline, bytes(0), src(1), dst(2), threadgroup-mem(0).
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_buffer(1, src, 0);
    enc.set_buffer(2, dst, 0);
    enc.set_threadgroup_memory_size(SMEM, 0);

    // ref: ggml-metal-ops.cpp:4197 — dispatch(nrows, 1, 1, nth, 1, 1).
    // nrows = ne01*ne02*ne03 (1-D src gets unrolled across the higher dims).
    let nrows = (ne01 * ne02 * ne03) as usize;
    enc.dispatch_threadgroups((nrows, 1, 1), (nth as usize, 1, 1));

    true
}

#[cfg(test)]
mod tests {
    use super::{mul_mv_shape, GgmlType};

    #[test]
    fn q5k_mul_mv_shape_matches_ggml_metal_constants() {
        let (shape, suffix) = mul_mv_shape(GgmlType::Q5_K, 1024);
        assert_eq!(shape.nsg, 2);
        assert_eq!(shape.nr0, 1);
        assert_eq!(shape.nr1, 1);
        assert_eq!(shape.smem, 0);
        assert_eq!(suffix, "");
    }
}
