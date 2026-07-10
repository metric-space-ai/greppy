// greppy-embed-native mean-pool + dispatch shader.
// Copyright (c) 2026 The greppy-rs authors. MIT License.
// Companion to the vendored ggml Metal kernels in this directory
// (ggml-metal.metal), Copyright (c) 2023-2026 The ggml authors, MIT License —
// see ../../../LICENSE-ggml.

#include <metal_stdlib>
using namespace metal;

struct embed_native_kargs_mean_pool {
    int32_t batch;
    int32_t seq_len;
    int32_t hidden;
};

struct embed_native_kargs_scale {
    int32_t n;
    float scale;
};

struct embed_native_kargs_rms_norm_f16 {
    int32_t ne00;
    int32_t ne01;
    int32_t ne02;
    int32_t ne03;
    uint64_t src_nb1;
    uint64_t src_nb2;
    uint64_t src_nb3;
    uint64_t dst_nb1;
    uint64_t dst_nb2;
    uint64_t dst_nb3;
    uint64_t add_nb1;
    uint64_t add_nb2;
    uint64_t add_nb3;
    float eps;
};

struct embed_native_kargs_geglu_f16 {
    int32_t rows;
    int32_t dim;
};

struct embed_native_kargs_rms_norm_rope {
    int32_t batch;
    int32_t seq_len;
    int32_t heads;
    int32_t head_dim;
    int32_t row_width;
    float eps;
    float freq_base;
    int32_t pad;
};

struct embed_native_kargs_post_attn_ffn_norm {
    int32_t rows;
    int32_t dim;
    float eps;
    int32_t pad;
};

struct embed_native_kargs_qwen_norm {
    int32_t rows;
    int32_t dim;
    float eps;
    int32_t qwen_scale;
};

struct embed_native_kargs_qwen_add {
    int32_t total;
};

struct embed_native_kargs_qwen_conv {
    int32_t channels;
    int32_t k_width;
};

struct embed_native_kargs_qwen_conv_rows {
    int32_t rows;
    int32_t channels;
    int32_t k_width;
    int32_t pad;
};

struct embed_native_kargs_qwen_heads {
    int32_t heads;
    int32_t head_dim;
    float eps;
    int32_t pad;
};

struct embed_native_kargs_qwen_heads_rows {
    int32_t rows;
    int32_t heads;
    int32_t head_dim;
    int32_t q_stride;
    int32_t k_stride;
    float eps;
    int32_t pad0;
    int32_t pad1;
};

struct embed_native_kargs_qwen_deltanet {
    int32_t heads;
    int32_t head_dim;
};

struct embed_native_kargs_qwen_deltanet_rows {
    int32_t rows;
    int32_t heads;
    int32_t head_dim;
    int32_t q_stride;
    int32_t k_stride;
    int32_t v_stride;
    int32_t beta_stride;
    int32_t alpha_stride;
    int32_t out_stride;
    int32_t pad0;
    int32_t pad1;
    int32_t pad2;
};

struct embed_native_kargs_qwen_rope {
    int32_t heads;
    int32_t head_dim;
    int32_t rope_dim;
    int32_t position;
    float base_freq;
    int32_t pad0;
    int32_t pad1;
    int32_t pad2;
};

struct embed_native_kargs_qwen_rope_rows {
    int32_t rows;
    int32_t heads;
    int32_t head_dim;
    int32_t rope_dim;
    int32_t position;
    int32_t stride;
    float base_freq;
    int32_t pad;
};

struct embed_native_kargs_qwen_cache {
    int32_t position;
    int32_t heads;
    int32_t head_dim;
    int32_t max_context;
};

struct embed_native_kargs_qwen_cache_rows {
    int32_t rows;
    int32_t position;
    int32_t heads;
    int32_t head_dim;
    int32_t max_context;
    int32_t src_stride;
    int32_t pad0;
    int32_t pad1;
};

struct embed_native_kargs_qwen_attn {
    int32_t position;
    int32_t q_heads;
    int32_t kv_heads;
    int32_t dim;
    int32_t max_context;
    float scale;
    int32_t pad0;
    int32_t pad1;
};

struct embed_native_kargs_qwen_attn_rows {
    int32_t rows;
    int32_t position;
    int32_t q_heads;
    int32_t kv_heads;
    int32_t dim;
    int32_t max_context;
    int32_t q_stride;
    int32_t score_stride;
    float scale;
    int32_t pad0;
    int32_t pad1;
    int32_t pad2;
};

struct embed_native_kargs_qwen_gate_rows {
    int32_t rows;
    int32_t width;
    int32_t value_stride;
    int32_t gate_stride;
};

kernel void embed_native_mean_pool_f32(
        constant embed_native_kargs_mean_pool & args [[buffer(0)]],
        device const float * hidden [[buffer(1)]],
        device const uint  * mask   [[buffer(2)]],
        device       float * dst    [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint b = gid.y;
    if (d >= (uint) args.hidden || b >= (uint) args.batch) {
        return;
    }

    float sum = 0.0f;
    float count = 0.0f;
    const uint row_base = b * args.seq_len;
    for (int s = 0; s < args.seq_len; ++s) {
        const float m = mask[row_base + s] == 0 ? 0.0f : 1.0f;
        count += m;
        sum += hidden[(row_base + s) * args.hidden + d] * m;
    }
    count = max(count, 1.0e-12f);
    dst[b * args.hidden + d] = sum / count;
}

kernel void embed_native_scale_f32_to_f16(
        constant embed_native_kargs_scale & args [[buffer(0)]],
        device const float * src [[buffer(1)]],
        device       half  * dst [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint) args.n) {
        return;
    }
    dst[gid] = half(src[gid] * args.scale);
}

kernel void embed_native_scale_f32(
        constant embed_native_kargs_scale & args [[buffer(0)]],
        device const float * src [[buffer(1)]],
        device       float * dst [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint) args.n) {
        return;
    }
    dst[gid] = src[gid] * args.scale;
}

kernel void embed_native_mean_pool_f16_to_f32(
        constant embed_native_kargs_mean_pool & args [[buffer(0)]],
        device const half * hidden [[buffer(1)]],
        device const uint * mask   [[buffer(2)]],
        device      float * dst    [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint b = gid.y;
    if (d >= (uint) args.hidden || b >= (uint) args.batch) {
        return;
    }

    float sum = 0.0f;
    float count = 0.0f;
    const uint row_base = b * args.seq_len;
    for (int s = 0; s < args.seq_len; ++s) {
        const float m = mask[row_base + s] == 0 ? 0.0f : 1.0f;
        count += m;
        sum += float(hidden[(row_base + s) * args.hidden + d]) * m;
    }
    count = max(count, 1.0e-12f);
    dst[b * args.hidden + d] = sum / count;
}

kernel void embed_native_rms_norm_mul_f16(
        constant embed_native_kargs_rms_norm_f16 & args [[buffer(0)]],
        device const char  * src    [[buffer(1)]],
        device const float * weight [[buffer(2)]],
        device       char  * dst    [[buffer(3)]],
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint sgitg [[simdgroup_index_in_threadgroup]],
        uint tiisg [[thread_index_in_simdgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;
    device const half * x = (device const half *) (src + i03*args.src_nb3 + i02*args.src_nb2 + i01*args.src_nb1);

    float sumf = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        const float v = float(x[i00]);
        sumf += v * v;
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float scale = rsqrt(sumf / args.ne00 + args.eps);
    device half * y = (device half *) (dst + i03*args.dst_nb3 + i02*args.dst_nb2 + i01*args.dst_nb1);
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        y[i00] = half(float(x[i00]) * scale * weight[i00]);
    }
}

kernel void embed_native_rms_norm_mul_add_f16(
        constant embed_native_kargs_rms_norm_f16 & args [[buffer(0)]],
        device const char  * src    [[buffer(1)]],
        device const float * weight [[buffer(2)]],
        device const char  * add    [[buffer(3)]],
        device       char  * dst    [[buffer(4)]],
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint sgitg [[simdgroup_index_in_threadgroup]],
        uint tiisg [[thread_index_in_simdgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;
    device const half * x = (device const half *) (src + i03*args.src_nb3 + i02*args.src_nb2 + i01*args.src_nb1);

    float sumf = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        const float v = float(x[i00]);
        sumf += v * v;
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float scale = rsqrt(sumf / args.ne00 + args.eps);
    device const half * a = (device const half *) (add + i03*args.add_nb3 + i02*args.add_nb2 + i01*args.add_nb1);
    device half * y = (device half *) (dst + i03*args.dst_nb3 + i02*args.dst_nb2 + i01*args.dst_nb1);
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        y[i00] = half(float(x[i00]) * scale * weight[i00] + float(a[i00]));
    }
}

kernel void embed_native_geglu_f16(
        constant embed_native_kargs_geglu_f16 & args [[buffer(0)]],
        device const half * gate [[buffer(1)]],
        device const half * up   [[buffer(2)]],
        device       half * dst  [[buffer(3)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (row >= (uint) args.rows) {
        return;
    }
    const uint base = row * args.dim;
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    constexpr float gelu_coef_a = 0.044715f;
    for (uint i = tid; i < (uint) args.dim; i += ntg) {
        const float x0 = float(gate[base + i]);
        const float x1 = float(up[base + i]);
        const float gelu = 0.5f * x0 * (1.0f + precise::tanh(sqrt_2_over_pi * x0 * (1.0f + gelu_coef_a * x0 * x0)));
        dst[base + i] = half(gelu * x1);
    }
}

kernel void embed_native_rms_norm_rope_neox_f32(
        constant embed_native_kargs_rms_norm_rope & args [[buffer(0)]],
        device const float * src    [[buffer(1)]],
        device const float * weight [[buffer(2)]],
        device       float * dst    [[buffer(3)]],
        threadgroup float * shmem   [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint3 ntg   [[threads_per_threadgroup]]) {
    const uint pos = tgpig.x;
    const uint head = tgpig.y;
    const uint batch = tgpig.z;
    const uint tid = tpitg.x;
    const uint nthreads = ntg.x;
    if (batch >= (uint) args.batch || pos >= (uint) args.seq_len || head >= (uint) args.heads) {
        return;
    }

    const uint head_dim = (uint) args.head_dim;
    const uint half_dim = head_dim / 2;
    const uint64_t src_base = ((uint64_t) batch * args.seq_len + pos) * args.row_width + head * head_dim;
    const uint64_t dst_base = src_base;

    float sumf = 0.0f;
    for (uint i = tid; i < head_dim; i += nthreads) {
        const float v = src[src_base + i];
        sumf += v * v;
    }
    shmem[tid] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv = rsqrt(shmem[0] / float(head_dim) + args.eps);
    for (uint i = tid; i < half_dim; i += nthreads) {
        const float x0 = src[src_base + i] * inv * weight[i];
        const float x1 = src[src_base + half_dim + i] * inv * weight[half_dim + i];
        const float theta = float(pos) * pow(args.freq_base, -2.0f * float(i) / float(head_dim));
        const float c = cos(theta);
        const float s = sin(theta);
        dst[dst_base + i] = x0 * c - x1 * s;
        dst[dst_base + half_dim + i] = x0 * s + x1 * c;
    }
}

kernel void embed_native_post_attn_ffn_norm_f32(
        constant embed_native_kargs_post_attn_ffn_norm & args [[buffer(0)]],
        device const float * attn_proj        [[buffer(1)]],
        device const float * residual         [[buffer(2)]],
        device const float * post_attn_weight [[buffer(3)]],
        device const float * ffn_weight       [[buffer(4)]],
        device       float * sa_out           [[buffer(5)]],
        device       float * ffn_norm         [[buffer(6)]],
        threadgroup float * shmem             [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (row >= (uint) args.rows) {
        return;
    }
    const uint64_t base = (uint64_t) row * args.dim;

    float sumf = 0.0f;
    for (uint i = tid; i < (uint) args.dim; i += ntg) {
        const float v = attn_proj[base + i];
        sumf += v * v;
    }
    shmem[tid] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float post_scale = rsqrt(shmem[0] / float(args.dim) + args.eps);

    sumf = 0.0f;
    for (uint i = tid; i < (uint) args.dim; i += ntg) {
        const float y = attn_proj[base + i] * post_scale * post_attn_weight[i] + residual[base + i];
        sa_out[base + i] = y;
        sumf += y * y;
    }
    shmem[tid] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float ffn_scale = rsqrt(shmem[0] / float(args.dim) + args.eps);
    for (uint i = tid; i < (uint) args.dim; i += ntg) {
        ffn_norm[base + i] = sa_out[base + i] * ffn_scale * ffn_weight[i];
    }
}

kernel void embed_native_post_ffn_next_attn_norm_f32(
        constant embed_native_kargs_post_attn_ffn_norm & args [[buffer(0)]],
        device const float * ffn_down         [[buffer(1)]],
        device const float * residual         [[buffer(2)]],
        device const float * post_ffn_weight  [[buffer(3)]],
        device const float * next_attn_weight [[buffer(4)]],
        device       float * out_state        [[buffer(5)]],
        device       float * next_attn_norm   [[buffer(6)]],
        threadgroup float * shmem             [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (row >= (uint) args.rows) {
        return;
    }
    const uint64_t base = (uint64_t) row * args.dim;

    float sumf = 0.0f;
    for (uint i = tid; i < (uint) args.dim; i += ntg) {
        const float v = ffn_down[base + i];
        sumf += v * v;
    }
    shmem[tid] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float post_scale = rsqrt(shmem[0] / float(args.dim) + args.eps);

    sumf = 0.0f;
    for (uint i = tid; i < (uint) args.dim; i += ntg) {
        const float y = ffn_down[base + i] * post_scale * post_ffn_weight[i] + residual[base + i];
        out_state[base + i] = y;
        sumf += y * y;
    }
    shmem[tid] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float next_scale = rsqrt(shmem[0] / float(args.dim) + args.eps);
    for (uint i = tid; i < (uint) args.dim; i += ntg) {
        next_attn_norm[base + i] = out_state[base + i] * next_scale * next_attn_weight[i];
    }
}

kernel void embed_native_qwen_rms_norm_f32(
        constant embed_native_kargs_qwen_norm & args [[buffer(0)]],
        device const float * src    [[buffer(1)]],
        device const float * weight [[buffer(2)]],
        device       float * dst    [[buffer(3)]],
        threadgroup float * shmem   [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (row >= (uint) args.rows) {
        return;
    }
    const uint64_t base = (uint64_t) row * args.dim;
    float sumf = 0.0f;
    for (uint i = tid; i < (uint) args.dim; i += ntg) {
        const float v = src[base + i];
        sumf += v * v;
    }
    shmem[tid] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = rsqrt(shmem[0] / float(args.dim) + args.eps);
    for (uint i = tid; i < (uint) args.dim; i += ntg) {
        dst[base + i] = src[base + i] * inv * weight[i];
    }
}

kernel void embed_native_qwen_add_rms_norm_f32(
        constant embed_native_kargs_qwen_norm & args [[buffer(0)]],
        device const float * lhs     [[buffer(1)]],
        device const float * rhs     [[buffer(2)]],
        device const float * weight  [[buffer(3)]],
        device       float * sum_out [[buffer(4)]],
        device       float * norm_out [[buffer(5)]],
        threadgroup float * shmem    [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (row >= (uint) args.rows) {
        return;
    }
    const uint64_t base = (uint64_t) row * args.dim;
    float sumf = 0.0f;
    for (uint i = tid; i < (uint) args.dim; i += ntg) {
        const float v = lhs[base + i] + rhs[base + i];
        sumf += v * v;
    }
    shmem[tid] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = rsqrt(shmem[0] / float(args.dim) + args.eps);
    for (uint i = tid; i < (uint) args.dim; i += ntg) {
        const float v = lhs[base + i] + rhs[base + i];
        sum_out[base + i] = v;
        norm_out[base + i] = v * inv * weight[i];
    }
}

// Qwen fixed-width normalization. One SIMDgroup owns a row, avoiding the
// 256-thread scratch reduction and its barriers on Apple GPUs.
kernel void embed_native_qwen_rms_norm_simd32_f32(
        constant embed_native_kargs_qwen_norm & args [[buffer(0)]],
        device const float * src    [[buffer(1)]],
        device const float * weight [[buffer(2)]],
        device       float * dst    [[buffer(3)]],
        uint row [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    if (row >= (uint) args.rows) {
        return;
    }
    const uint64_t base = (uint64_t) row * args.dim;
    float sumf = 0.0f;
    for (uint i = lane; i < (uint) args.dim; i += 32u) {
        const float v = src[base + i];
        sumf += v * v;
    }
    const float inv = rsqrt(simd_sum(sumf) / float(args.dim) + args.eps);
    for (uint i = lane; i < (uint) args.dim; i += 32u) {
        dst[base + i] = src[base + i] * inv * weight[i];
    }
}

kernel void embed_native_qwen_add_rms_norm_simd32_f32(
        constant embed_native_kargs_qwen_norm & args [[buffer(0)]],
        device const float * lhs      [[buffer(1)]],
        device const float * rhs      [[buffer(2)]],
        device const float * weight   [[buffer(3)]],
        device       float * sum_out  [[buffer(4)]],
        device       float * norm_out [[buffer(5)]],
        uint row [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    if (row >= (uint) args.rows) {
        return;
    }
    const uint64_t base = (uint64_t) row * args.dim;
    float sumf = 0.0f;
    for (uint i = lane; i < (uint) args.dim; i += 32u) {
        const float v = lhs[base + i] + rhs[base + i];
        sumf += v * v;
    }
    const float inv = rsqrt(simd_sum(sumf) / float(args.dim) + args.eps);
    for (uint i = lane; i < (uint) args.dim; i += 32u) {
        const float v = lhs[base + i] + rhs[base + i];
        sum_out[base + i] = v;
        norm_out[base + i] = v * inv * weight[i];
    }
}

kernel void embed_native_qwen_swiglu_f32(
        constant embed_native_kargs_qwen_add & args [[buffer(0)]],
        device const float * gate [[buffer(1)]],
        device const float * up   [[buffer(2)]],
        device       float * dst  [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint) args.total) {
        return;
    }
    const float x = gate[gid];
    dst[gid] = (x / (1.0f + exp(-x))) * up[gid];
}

kernel void embed_native_qwen_apply_silu_gate_f32(
        constant embed_native_kargs_qwen_add & args [[buffer(0)]],
        device       float * values [[buffer(1)]],
        device const float * gate   [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint) args.total) {
        return;
    }
    const float x = gate[gid];
    values[gid] *= x / (1.0f + exp(-x));
}

kernel void embed_native_qwen_apply_sigmoid_gate_f32(
        constant embed_native_kargs_qwen_add & args [[buffer(0)]],
        device       float * values [[buffer(1)]],
        device const float * gate   [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint) args.total) {
        return;
    }
    values[gid] *= 1.0f / (1.0f + exp(-gate[gid]));
}

kernel void embed_native_qwen_add_f32(
        constant embed_native_kargs_qwen_add & args [[buffer(0)]],
        device const float * lhs [[buffer(1)]],
        device const float * rhs [[buffer(2)]],
        device       float * dst [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint) args.total) {
        return;
    }
    dst[gid] = lhs[gid] + rhs[gid];
}

kernel void embed_native_qwen_causal_conv1d_silu_f32(
        constant embed_native_kargs_qwen_conv & args [[buffer(0)]],
        device       float * values [[buffer(1)]],
        device const float * weight [[buffer(2)]],
        device       float * state  [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    const int ch = int(gid);
    if (ch >= args.channels) {
        return;
    }
    const uint64_t base = (uint64_t) ch * args.k_width;
    for (int i = 0; i < args.k_width - 1; ++i) {
        state[base + i] = state[base + i + 1];
    }
    state[base + args.k_width - 1] = values[ch];
    float acc = 0.0f;
    for (int i = 0; i < args.k_width; ++i) {
        acc += state[base + i] * weight[base + i];
    }
    values[ch] = acc / (1.0f + exp(-acc));
}

kernel void embed_native_qwen_causal_conv1d_silu_rows_f32(
        constant embed_native_kargs_qwen_conv_rows & args [[buffer(0)]],
        device       float * values [[buffer(1)]],
        device const float * weight [[buffer(2)]],
        device       float * state  [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    const int ch = int(gid);
    if (ch >= args.channels) {
        return;
    }
    const uint64_t base = (uint64_t) ch * args.k_width;
    for (int row = 0; row < args.rows; ++row) {
        const uint64_t value_idx = (uint64_t) row * args.channels + ch;
        for (int i = 0; i < args.k_width - 1; ++i) {
            state[base + i] = state[base + i + 1];
        }
        state[base + args.k_width - 1] = values[value_idx];
        float acc = 0.0f;
        for (int i = 0; i < args.k_width; ++i) {
            acc += state[base + i] * weight[base + i];
        }
        values[value_idx] = acc / (1.0f + exp(-acc));
    }
}

kernel void embed_native_qwen_normalize_linear_qk_f32(
        constant embed_native_kargs_qwen_heads & args [[buffer(0)]],
        device float * q [[buffer(1)]],
        device float * k [[buffer(2)]],
        threadgroup float * shmem [[threadgroup(0)]],
        uint head [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (head >= (uint) args.heads) {
        return;
    }
    threadgroup float * sq = shmem;
    threadgroup float * sk = shmem + ntg;
    const uint64_t base = (uint64_t) head * args.head_dim;
    float sum_q = 0.0f;
    float sum_k = 0.0f;
    for (uint i = tid; i < (uint) args.head_dim; i += ntg) {
        const float qv = q[base + i];
        const float kv = k[base + i];
        sum_q += qv * qv;
        sum_k += kv * kv;
    }
    sq[tid] = sum_q;
    sk[tid] = sum_k;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sq[tid] += sq[tid + stride];
            sk[tid] += sk[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float q_scale = rsqrt(sq[0] + args.eps) * rsqrt(float(args.head_dim));
    const float k_scale = rsqrt(sk[0] + args.eps);
    for (uint i = tid; i < (uint) args.head_dim; i += ntg) {
        q[base + i] *= q_scale;
        k[base + i] *= k_scale;
    }
}

kernel void embed_native_qwen_normalize_linear_qk_rows_f32(
        constant embed_native_kargs_qwen_heads_rows & args [[buffer(0)]],
        device float * q [[buffer(1)]],
        device float * k [[buffer(2)]],
        threadgroup float * shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    const uint tid = tpitg.x;
    const uint nthreads = ntg.x;
    const uint row = tgpig.x;
    const uint head = tgpig.y;
    if (row >= (uint) args.rows || head >= (uint) args.heads) {
        return;
    }
    threadgroup float * sq = shmem;
    threadgroup float * sk = shmem + nthreads;
    const uint64_t q_base = (uint64_t) row * args.q_stride + (uint64_t) head * args.head_dim;
    const uint64_t k_base = (uint64_t) row * args.k_stride + (uint64_t) head * args.head_dim;
    float sum_q = 0.0f;
    float sum_k = 0.0f;
    for (uint i = tid; i < (uint) args.head_dim; i += nthreads) {
        const float qv = q[q_base + i];
        const float kv = k[k_base + i];
        sum_q += qv * qv;
        sum_k += kv * kv;
    }
    sq[tid] = sum_q;
    sk[tid] = sum_k;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sq[tid] += sq[tid + stride];
            sk[tid] += sk[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float q_scale = rsqrt(sq[0] + args.eps) * rsqrt(float(args.head_dim));
    const float k_scale = rsqrt(sk[0] + args.eps);
    for (uint i = tid; i < (uint) args.head_dim; i += nthreads) {
        q[q_base + i] *= q_scale;
        k[k_base + i] *= k_scale;
    }
}

kernel void embed_native_qwen_normalize_linear_qk_simd32_f32(
        constant embed_native_kargs_qwen_heads & args [[buffer(0)]],
        device float * q [[buffer(1)]],
        device float * k [[buffer(2)]],
        uint head [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    if (head >= (uint) args.heads) {
        return;
    }
    const uint64_t base = (uint64_t) head * args.head_dim;
    float sum_q = 0.0f;
    float sum_k = 0.0f;
    for (uint i = lane; i < (uint) args.head_dim; i += 32u) {
        const float qv = q[base + i];
        const float kv = k[base + i];
        sum_q += qv * qv;
        sum_k += kv * kv;
    }
    const float q_scale = rsqrt(simd_sum(sum_q) + args.eps) * rsqrt(float(args.head_dim));
    const float k_scale = rsqrt(simd_sum(sum_k) + args.eps);
    for (uint i = lane; i < (uint) args.head_dim; i += 32u) {
        q[base + i] *= q_scale;
        k[base + i] *= k_scale;
    }
}

kernel void embed_native_qwen_normalize_linear_qk_rows_simd32_f32(
        constant embed_native_kargs_qwen_heads_rows & args [[buffer(0)]],
        device float * q [[buffer(1)]],
        device float * k [[buffer(2)]],
        uint2 tg_pos [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg_pos.x;
    const uint head = tg_pos.y;
    if (row >= (uint) args.rows || head >= (uint) args.heads) {
        return;
    }
    const uint64_t q_base = (uint64_t) row * args.q_stride + (uint64_t) head * args.head_dim;
    const uint64_t k_base = (uint64_t) row * args.k_stride + (uint64_t) head * args.head_dim;
    float sum_q = 0.0f;
    float sum_k = 0.0f;
    for (uint i = lane; i < (uint) args.head_dim; i += 32u) {
        const float qv = q[q_base + i];
        const float kv = k[k_base + i];
        sum_q += qv * qv;
        sum_k += kv * kv;
    }
    const float q_scale = rsqrt(simd_sum(sum_q) + args.eps) * rsqrt(float(args.head_dim));
    const float k_scale = rsqrt(simd_sum(sum_k) + args.eps);
    for (uint i = lane; i < (uint) args.head_dim; i += 32u) {
        q[q_base + i] *= q_scale;
        k[k_base + i] *= k_scale;
    }
}

kernel void embed_native_qwen_deinterleave_q_gate_rows_f32(
        constant embed_native_kargs_qwen_heads_rows & args [[buffer(0)]],
        device const float * packed [[buffer(1)]],
        device       float * q_out  [[buffer(2)]],
        device       float * gate_out [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    const int per_row = args.heads * args.head_dim;
    const int idx = int(gid);
    if (idx >= args.rows * per_row) {
        return;
    }
    const int row = idx / per_row;
    const int local = idx - row * per_row;
    const int head = local / args.head_dim;
    const int lane = local - head * args.head_dim;
    const uint64_t src =
        (uint64_t) row * args.q_stride + (uint64_t) head * args.head_dim * 2 + lane;
    const uint64_t dst = (uint64_t) row * args.k_stride + local;
    q_out[dst] = packed[src];
    gate_out[dst] = packed[src + args.head_dim];
}

kernel void embed_native_qwen_rms_norm_strided_rows_f32(
        constant embed_native_kargs_qwen_heads_rows & args [[buffer(0)]],
        device float * values [[buffer(1)]],
        device const float * weight [[buffer(2)]],
        threadgroup float * shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    const uint tid = tpitg.x;
    const uint nthreads = ntg.x;
    const uint row = tgpig.x;
    const uint head = tgpig.y;
    if (row >= (uint) args.rows || head >= (uint) args.heads) {
        return;
    }
    const uint64_t base = (uint64_t) row * args.q_stride + (uint64_t) head * args.head_dim;
    float sumf = 0.0f;
    for (uint i = tid; i < (uint) args.head_dim; i += nthreads) {
        const float v = values[base + i];
        sumf += v * v;
    }
    shmem[tid] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float scale = rsqrt(shmem[0] / float(args.head_dim) + args.eps);
    for (uint i = tid; i < (uint) args.head_dim; i += nthreads) {
        values[base + i] *= scale * weight[i];
    }
}

kernel void embed_native_qwen_rms_norm_strided_rows_simd32_f32(
        constant embed_native_kargs_qwen_heads_rows & args [[buffer(0)]],
        device float * values [[buffer(1)]],
        device const float * weight [[buffer(2)]],
        uint2 tg_pos [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg_pos.x;
    const uint head = tg_pos.y;
    if (row >= (uint) args.rows || head >= (uint) args.heads) {
        return;
    }
    const uint64_t base = (uint64_t) row * args.q_stride + (uint64_t) head * args.head_dim;
    float sumf = 0.0f;
    for (uint i = lane; i < (uint) args.head_dim; i += 32u) {
        const float v = values[base + i];
        sumf += v * v;
    }
    const float scale = rsqrt(simd_sum(sumf) / float(args.head_dim) + args.eps);
    for (uint i = lane; i < (uint) args.head_dim; i += 32u) {
        values[base + i] *= scale * weight[i];
    }
}

kernel void embed_native_qwen_deltanet_decode_f32(
        constant embed_native_kargs_qwen_deltanet & args [[buffer(0)]],
        device const float * q       [[buffer(1)]],
        device const float * k       [[buffer(2)]],
        device const float * v       [[buffer(3)]],
        device const float * beta    [[buffer(4)]],
        device const float * alpha   [[buffer(5)]],
        device const float * a_log   [[buffer(6)]],
        device const float * dt_bias [[buffer(7)]],
        device       float * state   [[buffer(8)]],
        device       float * out     [[buffer(9)]],
        threadgroup float * shmem    [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    const uint tid = tpitg.x;
    const uint nthreads = ntg.x;
    const int head = int(tgpig.x);
    const int value_idx = int(tgpig.y);
    if (head >= args.heads || value_idx >= args.head_dim) {
        return;
    }
    threadgroup float * sprior = shmem;
    threadgroup float * sattn = shmem + nthreads;
    const uint64_t head_base = (uint64_t) head * args.head_dim;
    const uint64_t row_base = ((uint64_t) head * args.head_dim + value_idx) * args.head_dim;
    const float beta_h = 1.0f / (1.0f + exp(-beta[head]));
    const float x = alpha[head] + dt_bias[head];
    const float sp = x > 20.0f ? x : log(1.0f + exp(x));
    float decay = exp(-exp(a_log[head]) * sp);
    decay = min(max(decay, 0.0f), 1.0f);

    float prior = 0.0f;
    for (uint key_idx = tid; key_idx < (uint) args.head_dim; key_idx += nthreads) {
        prior += state[row_base + key_idx] * k[head_base + key_idx];
    }
    sprior[tid] = prior;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sprior[tid] += sprior[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float delta = (v[head_base + value_idx] - decay * sprior[0]) * beta_h;

    float attn = 0.0f;
    for (uint key_idx = tid; key_idx < (uint) args.head_dim; key_idx += nthreads) {
        const uint64_t idx = row_base + key_idx;
        const float updated = decay * state[idx] + k[head_base + key_idx] * delta;
        state[idx] = updated;
        attn += updated * q[head_base + key_idx];
    }
    sattn[tid] = attn;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sattn[tid] += sattn[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        out[head_base + value_idx] = sattn[0];
    }
}

kernel void embed_native_qwen_deltanet_decode_rows_f32(
        constant embed_native_kargs_qwen_deltanet_rows & args [[buffer(0)]],
        device const float * q       [[buffer(1)]],
        device const float * k       [[buffer(2)]],
        device const float * v       [[buffer(3)]],
        device const float * beta    [[buffer(4)]],
        device const float * alpha   [[buffer(5)]],
        device const float * a_log   [[buffer(6)]],
        device const float * dt_bias [[buffer(7)]],
        device       float * state   [[buffer(8)]],
        device       float * out     [[buffer(9)]],
        threadgroup float * shmem    [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    const uint tid = tpitg.x;
    const uint nthreads = ntg.x;
    const int head = int(tgpig.x);
    const int value_idx = int(tgpig.y);
    if (head >= args.heads || value_idx >= args.head_dim) {
        return;
    }
    threadgroup float * sprior = shmem;
    threadgroup float * sattn = shmem + nthreads;
    const uint64_t head_base = (uint64_t) head * args.head_dim;
    const uint64_t row_base = ((uint64_t) head * args.head_dim + value_idx) * args.head_dim;
    for (int row = 0; row < args.rows; ++row) {
        const uint64_t q_base = (uint64_t) row * args.q_stride + head_base;
        const uint64_t k_base = (uint64_t) row * args.k_stride + head_base;
        const uint64_t v_base = (uint64_t) row * args.v_stride + head_base;
        const uint64_t beta_base = (uint64_t) row * args.beta_stride;
        const uint64_t alpha_base = (uint64_t) row * args.alpha_stride;
        const float beta_h = 1.0f / (1.0f + exp(-beta[beta_base + head]));
        const float x = alpha[alpha_base + head] + dt_bias[head];
        const float sp = x > 20.0f ? x : log(1.0f + exp(x));
        float decay = exp(-exp(a_log[head]) * sp);
        decay = min(max(decay, 0.0f), 1.0f);

        float prior = 0.0f;
        for (uint key_idx = tid; key_idx < (uint) args.head_dim; key_idx += nthreads) {
            prior += state[row_base + key_idx] * k[k_base + key_idx];
        }
        sprior[tid] = prior;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = nthreads >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sprior[tid] += sprior[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float delta = (v[v_base + value_idx] - decay * sprior[0]) * beta_h;

        float attn = 0.0f;
        for (uint key_idx = tid; key_idx < (uint) args.head_dim; key_idx += nthreads) {
            const uint64_t idx = row_base + key_idx;
            const float updated = decay * state[idx] + k[k_base + key_idx] * delta;
            state[idx] = updated;
            attn += updated * q[q_base + key_idx];
        }
        sattn[tid] = attn;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = nthreads >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sattn[tid] += sattn[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            out[(uint64_t) row * args.out_stride + head_base + value_idx] = sattn[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Qwen3.5 H16xD128 prefill scan. One SIMDgroup owns 32 value rows and
// keeps each complete recurrent row local across the token sequence.
kernel void embed_native_qwen_deltanet_prefill_rowcache_block32_f32(
        constant embed_native_kargs_qwen_deltanet_rows & args [[buffer(0)]],
        device const float * q       [[buffer(1)]],
        device const float * k       [[buffer(2)]],
        device const float * v       [[buffer(3)]],
        device const float * beta    [[buffer(4)]],
        device const float * alpha   [[buffer(5)]],
        device const float * a_log   [[buffer(6)]],
        device const float * dt_bias [[buffer(7)]],
        device       float * state   [[buffer(8)]],
        device       float * out     [[buffer(9)]],
        uint2 tg_pos [[threadgroup_position_in_grid]],
        uint2 tid_pos [[thread_position_in_threadgroup]]) {
    constexpr uint head_dim = 128;
    constexpr uint rows_per_tg = 32;
    const uint lane = tid_pos.x;
    const uint row_block = tg_pos.x;
    const uint head = tg_pos.y;
    const uint value_idx = row_block * rows_per_tg + lane;
    if (head >= (uint) args.heads || value_idx >= head_dim) {
        return;
    }

    threadgroup float q_s[head_dim];
    threadgroup float k_s[head_dim];
    threadgroup float beta_s;
    threadgroup float decay_s;
    thread float row_state[head_dim];

    const uint64_t head_base = (uint64_t) head * head_dim;
    const uint64_t state_base = ((uint64_t) head * head_dim + value_idx) * head_dim;
    for (uint col = 0; col < head_dim; ++col) {
        row_state[col] = state[state_base + col];
    }

    for (int token = 0; token < args.rows; ++token) {
        const uint64_t q_base = (uint64_t) token * args.q_stride + head_base;
        const uint64_t k_base = (uint64_t) token * args.k_stride + head_base;
        const uint64_t v_base = (uint64_t) token * args.v_stride + head_base;
        for (uint col = lane; col < head_dim; col += rows_per_tg) {
            q_s[col] = q[q_base + col];
            k_s[col] = k[k_base + col];
        }
        if (lane == 0) {
            const uint64_t beta_base = (uint64_t) token * args.beta_stride;
            const uint64_t alpha_base = (uint64_t) token * args.alpha_stride;
            beta_s = 1.0f / (1.0f + exp(-beta[beta_base + head]));
            const float x = alpha[alpha_base + head] + dt_bias[head];
            const float sp = x > 20.0f ? x : log(1.0f + exp(x));
            decay_s = clamp(exp(-exp(a_log[head]) * sp), 0.0f, 1.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float prior = 0.0f;
        for (uint col = 0; col < head_dim; ++col) {
            prior += row_state[col] * k_s[col];
        }
        const float delta = (v[v_base + value_idx] - decay_s * prior) * beta_s;
        float attn = 0.0f;
        for (uint col = 0; col < head_dim; ++col) {
            const float updated = decay_s * row_state[col] + k_s[col] * delta;
            row_state[col] = updated;
            attn += updated * q_s[col];
        }
        out[(uint64_t) token * args.out_stride + head_base + value_idx] = attn;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint col = 0; col < head_dim; ++col) {
        state[state_base + col] = row_state[col];
    }
}

kernel void embed_native_qwen_rope_decode_f32(
        constant embed_native_kargs_qwen_rope & args [[buffer(0)]],
        device float * values [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    const int rope_half = args.rope_dim / 2;
    const int total = args.heads * rope_half;
    const int idx = int(gid);
    if (idx >= total) {
        return;
    }
    const int i = idx % rope_half;
    const int head = idx / rope_half;
    const uint64_t base = (uint64_t) head * args.head_dim;
    const float x1 = values[base + i];
    const float x2 = values[base + rope_half + i];
    const float inv = pow(args.base_freq, -2.0f * float(i) / float(args.rope_dim));
    const float theta = float(args.position) * inv;
    const float s = sin(theta);
    const float c = cos(theta);
    values[base + i] = x1 * c - x2 * s;
    values[base + rope_half + i] = x1 * s + x2 * c;
}

kernel void embed_native_qwen_rope_rows_f32(
        constant embed_native_kargs_qwen_rope_rows & args [[buffer(0)]],
        device float * values [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    const int rope_half = args.rope_dim / 2;
    const int per_row = args.heads * rope_half;
    const int idx = int(gid);
    if (idx >= args.rows * per_row) {
        return;
    }
    const int row = idx / per_row;
    const int local = idx - row * per_row;
    const int i = local % rope_half;
    const int head = local / rope_half;
    const uint64_t base = (uint64_t) row * args.stride + (uint64_t) head * args.head_dim;
    const float x1 = values[base + i];
    const float x2 = values[base + rope_half + i];
    const float inv = pow(args.base_freq, -2.0f * float(i) / float(args.rope_dim));
    const float theta = float(args.position + row) * inv;
    const float s = sin(theta);
    const float c = cos(theta);
    values[base + i] = x1 * c - x2 * s;
    values[base + rope_half + i] = x1 * s + x2 * c;
}

kernel void embed_native_qwen_cache_write_f32(
        constant embed_native_kargs_qwen_cache & args [[buffer(0)]],
        device const float * src   [[buffer(1)]],
        device       float * cache [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    const int total = args.heads * args.head_dim;
    const int idx = int(gid);
    if (idx >= total) {
        return;
    }
    cache[((uint64_t) args.position * args.heads * args.head_dim) + idx] = src[idx];
}

kernel void embed_native_qwen_cache_write_rows_f32(
        constant embed_native_kargs_qwen_cache_rows & args [[buffer(0)]],
        device const float * src   [[buffer(1)]],
        device       float * cache [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    const int row_width = args.heads * args.head_dim;
    const int idx = int(gid);
    if (idx >= args.rows * row_width) {
        return;
    }
    const int row = idx / row_width;
    const int local = idx - row * row_width;
    cache[((uint64_t) (args.position + row) * args.heads * args.head_dim) + local] =
        src[(uint64_t) row * args.src_stride + local];
}

kernel void embed_native_qwen_attention_scores_decode_f32(
        constant embed_native_kargs_qwen_attn & args [[buffer(0)]],
        device const float * q       [[buffer(1)]],
        device const float * k_cache [[buffer(2)]],
        device       float * scores  [[buffer(3)]],
        threadgroup float * shmem    [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    const uint tid = tpitg.x;
    const uint nthreads = ntg.x;
    const int pos = int(tgpig.x);
    const int q_head = int(tgpig.y);
    if (pos > args.position || q_head >= args.q_heads) {
        return;
    }
    const int gqa = args.q_heads / args.kv_heads;
    const int kv_head = q_head / gqa;
    const uint64_t q_base = (uint64_t) q_head * args.dim;
    const uint64_t k_base = ((uint64_t) pos * args.kv_heads + kv_head) * args.dim;
    float acc = 0.0f;
    for (uint i = tid; i < (uint) args.dim; i += nthreads) {
        acc += q[q_base + i] * k_cache[k_base + i];
    }
    shmem[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        scores[(uint64_t) q_head * args.max_context + pos] = shmem[0] * args.scale;
    }
}

kernel void embed_native_qwen_attention_scores_rows_f32(
        constant embed_native_kargs_qwen_attn_rows & args [[buffer(0)]],
        device const float * q       [[buffer(1)]],
        device const float * k_cache [[buffer(2)]],
        device       float * scores  [[buffer(3)]],
        threadgroup float * shmem    [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    const uint tid = tpitg.x;
    const uint nthreads = ntg.x;
    const int pos = int(tgpig.x);
    const int q_head = int(tgpig.y);
    const int row = int(tgpig.z);
    if (row >= args.rows || q_head >= args.q_heads || pos > args.position + row) {
        return;
    }
    const int gqa = args.q_heads / args.kv_heads;
    const int kv_head = q_head / gqa;
    const uint64_t q_base = (uint64_t) row * args.q_stride + (uint64_t) q_head * args.dim;
    const uint64_t k_base = ((uint64_t) pos * args.kv_heads + kv_head) * args.dim;
    float acc = 0.0f;
    for (uint i = tid; i < (uint) args.dim; i += nthreads) {
        acc += q[q_base + i] * k_cache[k_base + i];
    }
    shmem[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        scores[(uint64_t) row * args.score_stride + (uint64_t) q_head * args.max_context + pos] =
            shmem[0] * args.scale;
    }
}

kernel void embed_native_qwen_softmax_decode_f32(
        constant embed_native_kargs_qwen_attn & args [[buffer(0)]],
        device float * scores [[buffer(1)]],
        threadgroup float * shmem [[threadgroup(0)]],
        uint head [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (head >= (uint) args.q_heads) {
        return;
    }
    const uint64_t base = (uint64_t) head * args.max_context;
    float local_max = -INFINITY;
    for (uint pos = tid; pos <= (uint) args.position; pos += ntg) {
        local_max = max(local_max, scores[base + pos]);
    }
    shmem[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] = max(shmem[tid], shmem[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_v = shmem[0];
    float local_sum = 0.0f;
    for (uint pos = tid; pos <= (uint) args.position; pos += ntg) {
        const float v = exp(scores[base + pos] - max_v);
        scores[base + pos] = v;
        local_sum += v;
    }
    shmem[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = 1.0f / max(shmem[0], 1.0e-20f);
    for (uint pos = tid; pos <= (uint) args.position; pos += ntg) {
        scores[base + pos] *= inv;
    }
}

kernel void embed_native_qwen_softmax_rows_f32(
        constant embed_native_kargs_qwen_attn_rows & args [[buffer(0)]],
        device float * scores [[buffer(1)]],
        threadgroup float * shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    const uint tid = tpitg.x;
    const uint nthreads = ntg.x;
    const int row = int(tgpig.x);
    const int head = int(tgpig.y);
    if (row >= args.rows || head >= args.q_heads) {
        return;
    }
    const int position = args.position + row;
    const uint64_t base = (uint64_t) row * args.score_stride + (uint64_t) head * args.max_context;
    float local_max = -INFINITY;
    for (uint pos = tid; pos <= (uint) position; pos += nthreads) {
        local_max = max(local_max, scores[base + pos]);
    }
    shmem[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] = max(shmem[tid], shmem[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_v = shmem[0];
    float local_sum = 0.0f;
    for (uint pos = tid; pos <= (uint) position; pos += nthreads) {
        const float v = exp(scores[base + pos] - max_v);
        scores[base + pos] = v;
        local_sum += v;
    }
    shmem[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = 1.0f / max(shmem[0], 1.0e-20f);
    for (uint pos = tid; pos <= (uint) position; pos += nthreads) {
        scores[base + pos] *= inv;
    }
}

kernel void embed_native_qwen_attention_values_decode_f32(
        constant embed_native_kargs_qwen_attn & args [[buffer(0)]],
        device const float * scores  [[buffer(1)]],
        device const float * v_cache [[buffer(2)]],
        device       float * out     [[buffer(3)]],
        threadgroup float * shmem    [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    const uint tid = tpitg.x;
    const uint nthreads = ntg.x;
    const int q_head = int(tgpig.x);
    const int value_idx = int(tgpig.y);
    if (q_head >= args.q_heads || value_idx >= args.dim) {
        return;
    }
    const int gqa = args.q_heads / args.kv_heads;
    const int kv_head = q_head / gqa;
    const uint64_t score_base = (uint64_t) q_head * args.max_context;
    float acc = 0.0f;
    for (uint pos = tid; pos <= (uint) args.position; pos += nthreads) {
        const uint64_t v_base = ((uint64_t) pos * args.kv_heads + kv_head) * args.dim;
        acc += scores[score_base + pos] * v_cache[v_base + value_idx];
    }
    shmem[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        out[(uint64_t) q_head * args.dim + value_idx] = shmem[0];
    }
}

kernel void embed_native_qwen_attention_values_rows_f32(
        constant embed_native_kargs_qwen_attn_rows & args [[buffer(0)]],
        device const float * scores  [[buffer(1)]],
        device const float * v_cache [[buffer(2)]],
        device       float * out     [[buffer(3)]],
        threadgroup float * shmem    [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    const uint tid = tpitg.x;
    const uint nthreads = ntg.x;
    const int row = int(tgpig.x);
    const int q_head = int(tgpig.y);
    const int value_idx = int(tgpig.z);
    if (row >= args.rows || q_head >= args.q_heads || value_idx >= args.dim) {
        return;
    }
    const int position = args.position + row;
    const int gqa = args.q_heads / args.kv_heads;
    const int kv_head = q_head / gqa;
    const uint64_t score_base =
        (uint64_t) row * args.score_stride + (uint64_t) q_head * args.max_context;
    float acc = 0.0f;
    for (uint pos = tid; pos <= (uint) position; pos += nthreads) {
        const uint64_t v_base = ((uint64_t) pos * args.kv_heads + kv_head) * args.dim;
        acc += scores[score_base + pos] * v_cache[v_base + value_idx];
    }
    shmem[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        out[(uint64_t) row * args.q_heads * args.dim + (uint64_t) q_head * args.dim + value_idx] =
            shmem[0];
    }
}

// Qwen3.5 H8/KV2/D256 prefill attention. Each query head is owned by one
// SIMDgroup; every lane accumulates eight head dimensions without cross-SIMD
// reductions or threadgroup scratch.
kernel void embed_native_qwen_attention_scores_rows_simd32_f32(
        constant embed_native_kargs_qwen_attn_rows & args [[buffer(0)]],
        device const float * q       [[buffer(1)]],
        device const float * k_cache [[buffer(2)]],
        device       float * scores  [[buffer(3)]],
        uint2 tg_pos [[threadgroup_position_in_grid]],
        uint2 tid_pos [[thread_position_in_threadgroup]]) {
    const uint row = tg_pos.x;
    const uint q_head = tg_pos.y;
    const uint lane = tid_pos.x;
    if (row >= (uint) args.rows || q_head >= (uint) args.q_heads) {
        return;
    }
    const uint gqa = (uint) args.q_heads / (uint) args.kv_heads;
    const uint kv_head = q_head / gqa;
    const uint position = (uint) args.position + row;
    const uint64_t q_base = (uint64_t) row * args.q_stride + q_head * args.dim;
    const uint64_t score_base = (uint64_t) row * args.score_stride +
        (uint64_t) q_head * args.max_context;

    for (uint pos = 0; pos <= position; ++pos) {
        const uint64_t k_base = ((uint64_t) pos * args.kv_heads + kv_head) * args.dim;
        float dot = 0.0f;
        for (uint dim = lane; dim < (uint) args.dim; dim += 32u) {
            dot += q[q_base + dim] * k_cache[k_base + dim];
        }
        const float score = simd_sum(dot) * args.scale;
        if (lane == 0u) {
            scores[score_base + pos] = score;
        }
    }
}

kernel void embed_native_qwen_softmax_rows_simd32_f32(
        constant embed_native_kargs_qwen_attn_rows & args [[buffer(0)]],
        device float * scores [[buffer(1)]],
        uint2 tg_pos [[threadgroup_position_in_grid]],
        uint2 tid_pos [[thread_position_in_threadgroup]]) {
    const uint row = tg_pos.x;
    const uint head = tg_pos.y;
    const uint lane = tid_pos.x;
    if (row >= (uint) args.rows || head >= (uint) args.q_heads) {
        return;
    }
    const uint position = (uint) args.position + row;
    const uint64_t base = (uint64_t) row * args.score_stride +
        (uint64_t) head * args.max_context;
    float local_max = -INFINITY;
    for (uint pos = lane; pos <= position; pos += 32u) {
        local_max = max(local_max, scores[base + pos]);
    }
    const float max_value = simd_max(local_max);
    float local_sum = 0.0f;
    for (uint pos = lane; pos <= position; pos += 32u) {
        const float value = exp(scores[base + pos] - max_value);
        scores[base + pos] = value;
        local_sum += value;
    }
    const float inv_sum = 1.0f / max(simd_sum(local_sum), 1.0e-20f);
    for (uint pos = lane; pos <= position; pos += 32u) {
        scores[base + pos] *= inv_sum;
    }
}

kernel void embed_native_qwen_attention_values_gate_rows_simd32_f32(
        constant embed_native_kargs_qwen_attn_rows & args [[buffer(0)]],
        device const float * scores  [[buffer(1)]],
        device const float * v_cache [[buffer(2)]],
        device const float * gate    [[buffer(3)]],
        device       float * out     [[buffer(4)]],
        uint2 tg_pos [[threadgroup_position_in_grid]],
        uint2 tid_pos [[thread_position_in_threadgroup]]) {
    const uint row = tg_pos.x;
    const uint q_head = tg_pos.y;
    const uint lane = tid_pos.x;
    if (row >= (uint) args.rows || q_head >= (uint) args.q_heads) {
        return;
    }
    const uint gqa = (uint) args.q_heads / (uint) args.kv_heads;
    const uint kv_head = q_head / gqa;
    const uint position = (uint) args.position + row;
    const uint64_t score_base = (uint64_t) row * args.score_stride +
        (uint64_t) q_head * args.max_context;
    const uint64_t gate_base = (uint64_t) row * args.q_stride + q_head * args.dim;
    const uint64_t out_base = ((uint64_t) row * args.q_heads + q_head) * args.dim;

    for (uint dim = lane; dim < (uint) args.dim; dim += 32u) {
        float value = 0.0f;
        for (uint pos = 0; pos <= position; ++pos) {
            const uint64_t v_base = ((uint64_t) pos * args.kv_heads + kv_head) * args.dim;
            value += scores[score_base + pos] * v_cache[v_base + dim];
        }
        const float gate_value = gate[gate_base + dim];
        out[out_base + dim] = value / (1.0f + exp(-gate_value));
    }
}

// Fused H8/KV2/D256 prefill attention with online softmax. One SIMDgroup owns
// one query head; each lane retains eight value dimensions in registers.
kernel void embed_native_qwen_attention_fused_rows_simd32_f32(
        constant embed_native_kargs_qwen_attn_rows & args [[buffer(0)]],
        device const float * q       [[buffer(1)]],
        device const float * k_cache [[buffer(2)]],
        device const float * v_cache [[buffer(3)]],
        device const float * gate    [[buffer(4)]],
        device       float * out     [[buffer(5)]],
        uint2 tg_pos [[threadgroup_position_in_grid]],
        uint2 tid_pos [[thread_position_in_threadgroup]]) {
    const uint row = tg_pos.x;
    const uint q_head = tg_pos.y;
    const uint lane = tid_pos.x;
    if (row >= (uint) args.rows || q_head >= (uint) args.q_heads) {
        return;
    }
    const uint gqa = (uint) args.q_heads / (uint) args.kv_heads;
    const uint kv_head = q_head / gqa;
    const uint position = (uint) args.position + row;
    const uint64_t q_base = (uint64_t) row * args.q_stride + q_head * args.dim;
    const uint64_t gate_base = q_base;
    const uint64_t out_base = ((uint64_t) row * args.q_heads + q_head) * args.dim;
    float values[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float max_score = -INFINITY;
    float sum = 0.0f;

    for (uint pos = 0; pos <= position; ++pos) {
        const uint64_t k_base = ((uint64_t) pos * args.kv_heads + kv_head) * args.dim;
        float dot = 0.0f;
        for (uint dim = lane; dim < (uint) args.dim; dim += 32u) {
            dot += q[q_base + dim] * k_cache[k_base + dim];
        }
        const float score = simd_sum(dot) * args.scale;
        float next_max = 0.0f;
        float old_scale = 0.0f;
        float new_scale = 0.0f;
        if (lane == 0u) {
            next_max = max(max_score, score);
            old_scale = exp(max_score - next_max);
            new_scale = exp(score - next_max);
        }
        next_max = simd_broadcast(next_max, 0u);
        old_scale = simd_broadcast(old_scale, 0u);
        new_scale = simd_broadcast(new_scale, 0u);
        const uint64_t v_base = ((uint64_t) pos * args.kv_heads + kv_head) * args.dim;
        for (uint item = 0; item < 8u; ++item) {
            const uint dim = lane + item * 32u;
            values[item] = values[item] * old_scale + v_cache[v_base + dim] * new_scale;
        }
        sum = sum * old_scale + new_scale;
        max_score = next_max;
    }

    const float inv_sum = 1.0f / max(sum, 1.0e-20f);
    for (uint item = 0; item < 8u; ++item) {
        const uint dim = lane + item * 32u;
        const float gate_value = gate[gate_base + dim];
        out[out_base + dim] = values[item] * inv_sum / (1.0f + exp(-gate_value));
    }
}

kernel void embed_native_qwen_apply_silu_gate_rows_f32(
        constant embed_native_kargs_qwen_gate_rows & args [[buffer(0)]],
        device       float * values [[buffer(1)]],
        device const float * gate   [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    const int idx = int(gid);
    if (idx >= args.rows * args.width) {
        return;
    }
    const int row = idx / args.width;
    const int col = idx - row * args.width;
    const float x = gate[(uint64_t) row * args.gate_stride + col];
    values[(uint64_t) row * args.value_stride + col] *= x / (1.0f + exp(-x));
}

kernel void embed_native_qwen_apply_sigmoid_gate_rows_f32(
        constant embed_native_kargs_qwen_gate_rows & args [[buffer(0)]],
        device       float * values [[buffer(1)]],
        device const float * gate   [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    const int idx = int(gid);
    if (idx >= args.rows * args.width) {
        return;
    }
    const int row = idx / args.width;
    const int col = idx - row * args.width;
    const float x = gate[(uint64_t) row * args.gate_stride + col];
    values[(uint64_t) row * args.value_stride + col] *= 1.0f / (1.0f + exp(-x));
}
