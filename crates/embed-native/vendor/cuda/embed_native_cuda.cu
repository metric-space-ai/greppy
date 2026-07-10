// Source-built CUDA backend for greppy-embed-native.
//
// Copyright (c) 2026 The greppy-rs authors. MIT License.
// Derivative work of ggml (https://github.com/ggml-org/ggml), Copyright (c)
// 2023-2026 The ggml authors, MIT License — see ../LICENSE-ggml. This file
// includes and dispatches the vendored ggml CUDA kernels in ggml-cuda/.
//
// The quantized matmul paths intentionally route through ggml CUDA kernels:
// batched/prefill calls use MMQ (`mul_mat_q` + `quantize_mmq_q8_1`), while
// batch-1 decode calls use MMVQ (`mul_mat_vec_q` + `quantize_row_q8_1`).

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "mmq.cuh"
#include "quantize.cuh"

#if defined(_WIN32)
#define GP_CUDA_EXPORT extern "C" __declspec(dllexport)
#else
#define GP_CUDA_EXPORT extern "C" __attribute__((visibility("default")))
#endif

#define GP_CUDA_DRIVER_ERROR_BASE 20000
#define GP_CUDA_UNSUPPORTED_ARCH 30001

static ggml_cuda_device_info gp_cuda_info_state;
static int gp_cuda_current_device = 0;

const ggml_cuda_device_info & ggml_cuda_info() {
    return gp_cuda_info_state;
}

void ggml_cuda_set_device(int device) {
    gp_cuda_current_device = device;
    cudaSetDevice(device);
}

int ggml_cuda_get_device() {
    int device = gp_cuda_current_device;
    if (cudaGetDevice(&device) == cudaSuccess) {
        gp_cuda_current_device = device;
    }
    return gp_cuda_current_device;
}

extern "C" void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fprintf(stderr, "ggml_abort at %s:%d: ", file ? file : "<unknown>", line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    abort();
}

GP_CUDA_EXPORT const char * gp_cuda_error_string(int code) {
    static char driver_error_buf[256];
    if (code >= GP_CUDA_DRIVER_ERROR_BASE && code < GP_CUDA_DRIVER_ERROR_BASE + 10000) {
        const char * name = nullptr;
        const char * desc = nullptr;
        CUresult cu = (CUresult)(code - GP_CUDA_DRIVER_ERROR_BASE);
        cuGetErrorName(cu, &name);
        cuGetErrorString(cu, &desc);
        snprintf(driver_error_buf, sizeof(driver_error_buf), "%s%s%s",
                 name ? name : "CUDA driver error",
                 desc ? ": " : "",
                 desc ? desc : "");
        return driver_error_buf;
    }
    if (code == GP_CUDA_UNSUPPORTED_ARCH) {
        return "CUDA GPU compute capability is below sm_75; greppy native CUDA MMQ backend is unavailable";
    }
    if (code <= 999) {
        return cudaGetErrorString((cudaError_t) code);
    }
    switch (code) {
        case 10000 + CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case 10000 + CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case 10000 + CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case 10000 + CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case 10000 + CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case 10000 + CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case 10000 + CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case 10000 + CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        default: return "unknown CUDA/cuBLAS error";
    }
}

GP_CUDA_EXPORT int gp_cuda_init(int device, void ** stream_out, void ** blas_out) {
    CUresult cu = cuInit(0);
    if (cu != CUDA_SUCCESS) return GP_CUDA_DRIVER_ERROR_BASE + (int) cu;

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) return (int) err;

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) return (int) err;
    if (prop.major*10 + prop.minor < 75) {
        return GP_CUDA_UNSUPPORTED_ARCH;
    }
    size_t free_vram = 0;
    size_t total_vram = 0;
    cudaMemGetInfo(&free_vram, &total_vram);
    gp_cuda_current_device = device;
    gp_cuda_info_state.device_count = 1;
    gp_cuda_info_state.devices[device].cc = prop.major*100 + prop.minor*10;
    gp_cuda_info_state.devices[device].nsm = prop.multiProcessorCount;
    gp_cuda_info_state.devices[device].smpb = prop.sharedMemPerBlock;
    gp_cuda_info_state.devices[device].smpbo = prop.sharedMemPerBlockOptin;
    gp_cuda_info_state.devices[device].integrated = prop.integrated != 0;
    gp_cuda_info_state.devices[device].vmm = false;
    gp_cuda_info_state.devices[device].vmm_granularity = 0;
    gp_cuda_info_state.devices[device].total_vram = total_vram;
    gp_cuda_info_state.devices[device].warp_size = prop.warpSize;
    gp_cuda_info_state.devices[device].supports_cooperative_launch = prop.cooperativeLaunch != 0;

    cudaStream_t stream = nullptr;
    err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) return (int) err;

    cublasHandle_t blas = nullptr;
    cublasStatus_t st = cublasCreate(&blas);
    if (st != CUBLAS_STATUS_SUCCESS) {
        cudaStreamDestroy(stream);
        return 10000 + (int) st;
    }
    st = cublasSetStream(blas, stream);
    if (st != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(blas);
        cudaStreamDestroy(stream);
        return 10000 + (int) st;
    }
    cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH);
    *stream_out = (void *) stream;
    *blas_out = (void *) blas;
    return 0;
}

GP_CUDA_EXPORT int gp_cuda_graph_begin(void * stream) {
    return (int)cudaStreamBeginCapture((cudaStream_t)stream, cudaStreamCaptureModeThreadLocal);
}

GP_CUDA_EXPORT int gp_cuda_graph_end(void * stream, void ** graph_out, void ** exec_out) {
    cudaGraph_t graph = nullptr;
    cudaError_t err = cudaStreamEndCapture((cudaStream_t)stream, &graph);
    if (err != cudaSuccess) return (int)err;
    cudaGraphExec_t exec = nullptr;
    err = cudaGraphInstantiateWithFlags(&exec, graph, 0);
    if (err != cudaSuccess) {
        cudaGraphDestroy(graph);
        return (int)err;
    }
    *graph_out = (void *)graph;
    *exec_out = (void *)exec;
    return 0;
}

GP_CUDA_EXPORT int gp_cuda_graph_abort(void * stream) {
    cudaGraph_t graph = nullptr;
    const cudaError_t err = cudaStreamEndCapture((cudaStream_t)stream, &graph);
    if (graph != nullptr) cudaGraphDestroy(graph);
    return err == cudaErrorStreamCaptureInvalidated ? 0 : (int)err;
}

GP_CUDA_EXPORT int gp_cuda_graph_launch(void * exec, void * stream) {
    return (int)cudaGraphLaunch((cudaGraphExec_t)exec, (cudaStream_t)stream);
}

GP_CUDA_EXPORT int gp_cuda_graph_destroy(void * graph, void * exec) {
    cudaError_t first = cudaSuccess;
    if (exec != nullptr) first = cudaGraphExecDestroy((cudaGraphExec_t)exec);
    if (graph != nullptr) {
        const cudaError_t err = cudaGraphDestroy((cudaGraph_t)graph);
        if (first == cudaSuccess) first = err;
    }
    return (int)first;
}

extern "C" int64_t ggml_blck_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            return 1;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
            return 32;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_K:
            return 256;
        default:
            return 1;
    }
}

extern "C" size_t ggml_type_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:  return 2;
        case GGML_TYPE_BF16: return 2;
        case GGML_TYPE_Q4_0: return 18;
        case GGML_TYPE_Q4_1: return 20;
        case GGML_TYPE_Q5_0: return 22;
        case GGML_TYPE_Q5_1: return 24;
        case GGML_TYPE_Q8_0: return 34;
        case GGML_TYPE_Q8_1: return 36;
        case GGML_TYPE_Q2_K: return 84;
        case GGML_TYPE_Q3_K: return 110;
        case GGML_TYPE_Q4_K: return 144;
        case GGML_TYPE_Q5_K: return 176;
        case GGML_TYPE_Q6_K: return 210;
        case GGML_TYPE_Q8_K: return 292;
        default:             return 0;
    }
}

extern "C" size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    const int64_t blck = ggml_blck_size(type);
    if (blck <= 0 || ne < 0) {
        return 0;
    }
    return (size_t) ((ne + blck - 1) / blck) * ggml_type_size(type);
}

extern "C" double ggml_type_sizef(enum ggml_type type) {
    const int64_t blck = ggml_blck_size(type);
    if (blck <= 0) {
        return 0.0;
    }
    return (double) ggml_type_size(type) / (double) blck;
}

#define GP_STANDALONE_MMVQ
#include "mmvq.cu"

GP_CUDA_EXPORT int gp_cuda_destroy(void * stream, void * blas) {
    if (blas) {
        cublasDestroy((cublasHandle_t) blas);
    }
    if (stream) {
        cudaStreamDestroy((cudaStream_t) stream);
    }
    return 0;
}

GP_CUDA_EXPORT int gp_cuda_malloc(void ** ptr, size_t bytes) {
    return (int) cudaMalloc(ptr, bytes);
}

GP_CUDA_EXPORT int gp_cuda_free(void * ptr) {
    return (int) cudaFree(ptr);
}

GP_CUDA_EXPORT int gp_cuda_memcpy_h2d_async(void * dst, const void * src, size_t bytes, void * stream) {
    return (int) cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, (cudaStream_t) stream);
}

GP_CUDA_EXPORT int gp_cuda_memcpy_d2h_async(void * dst, const void * src, size_t bytes, void * stream) {
    return (int) cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, (cudaStream_t) stream);
}

GP_CUDA_EXPORT int gp_cuda_memcpy_d2d_async(void * dst, const void * src, size_t bytes, void * stream) {
    return (int) cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, (cudaStream_t) stream);
}

GP_CUDA_EXPORT int gp_cuda_memset_async(void * dst, int value, size_t bytes, void * stream) {
    return (int) cudaMemsetAsync(dst, value, bytes, (cudaStream_t) stream);
}

GP_CUDA_EXPORT int gp_cuda_stream_sync(void * stream) {
    return (int) cudaStreamSynchronize((cudaStream_t) stream);
}

GP_CUDA_EXPORT int gp_cuda_mem_get_info(size_t * free_out, size_t * total_out) {
    return (int) cudaMemGetInfo(free_out, total_out);
}

static __device__ __forceinline__ float gp_half_to_float(const half h) {
    return __half2float(h);
}

struct gp_block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};

struct gp_block_q4_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

static __device__ __forceinline__ void gp_get_scale_min_k4(
        int j,
        const uint8_t * __restrict__ q,
        uint8_t * sc,
        uint8_t * m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

__global__ void gp_embed_q4k_kernel(
        const gp_block_q4_K * __restrict__ weights,
        const uint32_t * __restrict__ ids,
        float * __restrict__ dst,
        int rows,
        int blocks_per_row,
        float scale) {
    const int row = blockIdx.x;
    const int block_col = blockIdx.y;
    const int t = threadIdx.x;
    if (row >= rows || t >= 256) return;

    const uint32_t token = ids[row];
    const gp_block_q4_K & b = weights[(int64_t) token*blocks_per_row + block_col];
    const int sub = t >> 5;
    const int lane = t & 31;
    const int qs_idx = (sub >> 1)*32 + lane;
    const uint8_t packed = b.qs[qs_idx];
    const uint8_t q = (sub & 1) == 0 ? (packed & 0x0f) : (packed >> 4);
    uint8_t sc = 0;
    uint8_t m = 0;
    gp_get_scale_min_k4(sub, b.scales, &sc, &m);
    const float d = gp_half_to_float(b.d);
    const float dmin = gp_half_to_float(b.dmin);
    dst[(int64_t) row*blocks_per_row*256 + block_col*256 + t] =
        scale * (d * (float) sc * (float) q - dmin * (float) m);
}

GP_CUDA_EXPORT int gp_embed_q4k(
        const void * weights,
        const uint32_t * ids,
        float * dst,
        int rows,
        int hidden,
        float scale,
        void * stream) {
    if (hidden <= 0 || hidden % 256 != 0 || rows <= 0) {
        return (int) cudaErrorInvalidValue;
    }
    const int blocks_per_row = hidden / 256;
    gp_embed_q4k_kernel<<<dim3(rows, blocks_per_row, 1), 256, 0, (cudaStream_t) stream>>>(
        (const gp_block_q4_K *) weights, ids, dst, rows, blocks_per_row, scale);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_embed_q6k_kernel(
        const gp_block_q6_K * __restrict__ weights,
        const uint32_t * __restrict__ ids,
        float * __restrict__ dst,
        int rows,
        int blocks_per_row,
        float scale) {
    const int row = blockIdx.x;
    const int block_col = blockIdx.y;
    const int t = threadIdx.x;
    if (row >= rows || t >= 256) return;

    const uint32_t token = ids[row];
    const gp_block_q6_K & b = weights[(int64_t) token*blocks_per_row + block_col];
    const int n = t < 128 ? 0 : 128;
    const int idx = n / 128;
    const uint8_t * ql = b.ql + 64*idx;
    const uint8_t * qh = b.qh + 32*idx;
    const int8_t * sc = b.scales + 8*idx;
    const int l = t - n;
    const int lane = l & 31;
    const int is = lane / 16;
    int q;
    int sidx;
    if (l < 32) {
        q = ((ql[lane] & 0x0f) | ((qh[lane] & 3) << 4)) - 32;
        sidx = is;
    } else if (l < 64) {
        q = ((ql[lane + 32] & 0x0f) | (((qh[lane] >> 2) & 3) << 4)) - 32;
        sidx = is + 2;
    } else if (l < 96) {
        q = ((ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4)) - 32;
        sidx = is + 4;
    } else {
        q = ((ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 3) << 4)) - 32;
        sidx = is + 6;
    }
    const float d = gp_half_to_float(b.d);
    dst[(int64_t) row*blocks_per_row*256 + block_col*256 + t] =
        scale * d * (float) sc[sidx] * (float) q;
}

GP_CUDA_EXPORT int gp_embed_q6k(
        const void * weights,
        const uint32_t * ids,
        float * dst,
        int rows,
        int hidden,
        float scale,
        void * stream) {
    const int blocks_per_row = hidden / 256;
    gp_embed_q6k_kernel<<<dim3(rows, blocks_per_row, 1), 256, 0, (cudaStream_t) stream>>>(
        (const gp_block_q6_K *) weights, ids, dst, rows, blocks_per_row, scale);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_rms_norm_kernel(
        const float * __restrict__ src,
        const float * __restrict__ weight,
        const float * __restrict__ add,
        float * __restrict__ dst,
        int rows,
        int dim,
        float eps,
        int add_one_to_weight) {
    extern __shared__ float s[];
    const int row = blockIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = src[(int64_t) row*dim + i];
        sum += v*v;
    }
    s[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(s[0] / (float) dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const int64_t idx = (int64_t) row*dim + i;
        float v = src[idx] * inv * weight[i];
        if (add) v += add[idx];
        dst[idx] = v;
    }
}

GP_CUDA_EXPORT int gp_rms_norm(
        const float * src, const float * weight, float * dst,
        int rows, int dim, float eps, void * stream) {
    gp_rms_norm_kernel<<<rows, 256, 256*sizeof(float), (cudaStream_t) stream>>>(
        src, weight, nullptr, dst, rows, dim, eps, 0);
    return (int) cudaPeekAtLastError();
}

GP_CUDA_EXPORT int gp_rms_norm_add(
        const float * src, const float * add, const float * weight, float * dst,
        int rows, int dim, float eps, void * stream) {
    gp_rms_norm_kernel<<<rows, 256, 256*sizeof(float), (cudaStream_t) stream>>>(
        src, weight, add, dst, rows, dim, eps, 0);
    return (int) cudaPeekAtLastError();
}

GP_CUDA_EXPORT int gp_qwen_rms_norm(
        const float * src, const float * weight, float * dst,
        int rows, int dim, float eps, void * stream) {
    gp_rms_norm_kernel<<<rows, 256, 256*sizeof(float), (cudaStream_t) stream>>>(
        src, weight, nullptr, dst, rows, dim, eps, 1);
    return (int) cudaPeekAtLastError();
}

GP_CUDA_EXPORT int gp_qwen_rms_norm_add(
        const float * src, const float * add, const float * weight, float * dst,
        int rows, int dim, float eps, void * stream) {
    gp_rms_norm_kernel<<<rows, 256, 256*sizeof(float), (cudaStream_t) stream>>>(
        src, weight, add, dst, rows, dim, eps, 1);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_qwen_add_rms_norm_kernel(
        const float * __restrict__ lhs,
        const float * __restrict__ rhs,
        const float * __restrict__ weight,
        float * __restrict__ sum_out,
        float * __restrict__ norm_out,
        int rows,
        int dim,
        float eps) {
    extern __shared__ float s[];
    const int row = blockIdx.x;
    if (row >= rows) return;
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const int64_t idx = (int64_t) row*dim + i;
        const float v = lhs[idx] + rhs[idx];
        sum_sq += v*v;
    }
    s[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(s[0] / (float) dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const int64_t idx = (int64_t) row*dim + i;
        const float v = lhs[idx] + rhs[idx];
        sum_out[idx] = v;
        norm_out[idx] = v * inv * weight[i];
    }
}

GP_CUDA_EXPORT int gp_qwen_add_rms_norm(
        const float * lhs, const float * rhs, const float * weight,
        float * sum_out, float * norm_out,
        int rows, int dim, float eps, void * stream) {
    gp_qwen_add_rms_norm_kernel<<<rows, 256, 256*sizeof(float), (cudaStream_t) stream>>>(
        lhs, rhs, weight, sum_out, norm_out, rows, dim, eps);
    return (int) cudaPeekAtLastError();
}

template <bool add_residual>
__global__ void gp_qwen_rms_norm_q8_kernel(
        const float * __restrict__ src,
        const float * __restrict__ residual,
        const float * __restrict__ weight,
        float * __restrict__ sum_out,
        void * __restrict__ q8_out,
        int dim,
        float eps) {
    extern __shared__ float reduce[];
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float value = src[i];
        if constexpr (add_residual) {
            value += residual[i];
            sum_out[i] = value;
        }
        sum_sq += value * value;
    }
    reduce[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv = rsqrtf(reduce[0] / (float)dim + eps);
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    block_q8_1 * q8 = (block_q8_1 *)q8_out;
    for (int block = warp; block < dim / QK8_1; block += blockDim.x / 32) {
        const int index = block * QK8_1 + lane;
        const float input = add_residual ? sum_out[index] : src[index];
        const float value = input * inv * weight[index];
        float amax = warp_reduce_max<QK8_1>(fabsf(value));
        float sum = warp_reduce_sum<QK8_1>(value);
        const float d = amax / 127.0f;
        q8[block].qs[lane] = amax == 0.0f ? 0 : roundf(value / d);
        if (lane == 0) q8[block].ds = make_half2(d, sum);
    }
}

GP_CUDA_EXPORT int gp_qwen_rms_norm_q8(
        const float * src,
        const float * weight,
        void * q8_out,
        int dim,
        float eps,
        void * stream) {
    if (dim <= 0 || dim % 256 != 0) return (int)cudaErrorInvalidValue;
    gp_qwen_rms_norm_q8_kernel<false><<<1, 256, 256 * sizeof(float),
        (cudaStream_t)stream>>>(src, nullptr, weight, nullptr, q8_out, dim, eps);
    return (int)cudaPeekAtLastError();
}

GP_CUDA_EXPORT int gp_qwen_add_rms_norm_q8(
        const float * lhs,
        const float * rhs,
        const float * weight,
        float * sum_out,
        void * q8_out,
        int dim,
        float eps,
        void * stream) {
    if (dim <= 0 || dim % 256 != 0) return (int)cudaErrorInvalidValue;
    gp_qwen_rms_norm_q8_kernel<true><<<1, 256, 256 * sizeof(float),
        (cudaStream_t)stream>>>(lhs, rhs, weight, sum_out, q8_out, dim, eps);
    return (int)cudaPeekAtLastError();
}

__global__ void gp_qwen_causal_conv1d_silu_kernel(
        float * __restrict__ values,
        const float * __restrict__ weights,
        float * __restrict__ state,
        int channels,
        int kernel) {
    const int ch = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (ch >= channels) return;
    const int64_t base = (int64_t) ch * kernel;
    for (int i = 0; i < kernel - 1; ++i) {
        state[base + i] = state[base + i + 1];
    }
    state[base + kernel - 1] = values[ch];
    float acc = 0.0f;
    for (int i = 0; i < kernel; ++i) {
        acc += state[base + i] * weights[base + i];
    }
    values[ch] = acc / (1.0f + expf(-acc));
}

GP_CUDA_EXPORT int gp_qwen_causal_conv1d_silu(
        float * values,
        const float * weights,
        float * state,
        int channels,
        int kernel,
        void * stream) {
    if (channels <= 0 || kernel <= 0) return (int) cudaErrorInvalidValue;
    const int block = 256;
    const int grid = (channels + block - 1) / block;
    gp_qwen_causal_conv1d_silu_kernel<<<grid, block, 0, (cudaStream_t) stream>>>(
        values, weights, state, channels, kernel);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_qwen_causal_conv1d_silu_rows_kernel(
        float * __restrict__ values,
        const float * __restrict__ weights,
        float * __restrict__ state,
        int rows,
        int channels,
        int kernel) {
    const int ch = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (ch >= channels) return;
    const int64_t base = (int64_t) ch * kernel;
    for (int row = 0; row < rows; ++row) {
        const int64_t value_idx = (int64_t) row * channels + ch;
        for (int i = 0; i < kernel - 1; ++i) {
            state[base + i] = state[base + i + 1];
        }
        state[base + kernel - 1] = values[value_idx];
        float acc = 0.0f;
        for (int i = 0; i < kernel; ++i) {
            acc += state[base + i] * weights[base + i];
        }
        values[value_idx] = acc / (1.0f + expf(-acc));
    }
}

GP_CUDA_EXPORT int gp_qwen_causal_conv1d_silu_rows(
        float * values,
        const float * weights,
        float * state,
        int rows,
        int channels,
        int kernel,
        void * stream) {
    if (rows <= 0 || channels <= 0 || kernel <= 0) return (int) cudaErrorInvalidValue;
    const int block = 256;
    const int grid = (channels + block - 1) / block;
    gp_qwen_causal_conv1d_silu_rows_kernel<<<grid, block, 0, (cudaStream_t) stream>>>(
        values, weights, state, rows, channels, kernel);
    return (int) cudaPeekAtLastError();
}

template <int kernel, int rows_per_block>
__global__ void gp_qwen_causal_conv1d_silu_rows_parallel_kernel(
        const float * __restrict__ values,
        const float * __restrict__ weights,
        const float * __restrict__ state,
        float * __restrict__ out,
        int rows,
        int channels) {
    const int ch = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if (ch >= channels) return;
    const int row_begin = (int)blockIdx.x * rows_per_block;
    const int row_end = min(row_begin + rows_per_block, rows);
    const int64_t state_base = (int64_t)ch * kernel;
    float old_state[kernel];
    float w[kernel];
#pragma unroll
    for (int tap = 0; tap < kernel; ++tap) {
        old_state[tap] = state[state_base + tap];
        w[tap] = weights[state_base + tap];
    }
    for (int row = row_begin; row < row_end; ++row) {
        float acc = 0.0f;
#pragma unroll
        for (int tap = 0; tap < kernel; ++tap) {
            const int source_row = row + tap - (kernel - 1);
            const float x = source_row < 0
                ? old_state[source_row + kernel]
                : values[(int64_t)source_row * channels + ch];
            acc += x * w[tap];
        }
        out[(int64_t)row * channels + ch] = acc / (1.0f + expf(-acc));
    }
}

template <int kernel>
__global__ void gp_qwen_causal_conv1d_update_state_kernel(
        const float * __restrict__ values,
        float * __restrict__ state,
        int rows,
        int channels) {
    const int ch = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (ch >= channels) return;
    const int64_t state_base = (int64_t)ch * kernel;
    float old_state[kernel];
#pragma unroll
    for (int tap = 0; tap < kernel; ++tap) {
        old_state[tap] = state[state_base + tap];
    }
#pragma unroll
    for (int tap = 0; tap < kernel; ++tap) {
        const int source_row = rows + tap - kernel;
        state[state_base + tap] = source_row < 0
            ? old_state[source_row + kernel]
            : values[(int64_t)source_row * channels + ch];
    }
}

GP_CUDA_EXPORT int gp_qwen_causal_conv1d_silu_rows_parallel(
        const float * values,
        const float * weights,
        float * state,
        float * out,
        int rows,
        int channels,
        int kernel,
        void * stream) {
    if (rows <= 0 || channels <= 0 || kernel != 4) return (int) cudaErrorInvalidValue;
    constexpr int threads = 128;
    constexpr int rows_per_block = 32;
    const dim3 grid((rows + rows_per_block - 1) / rows_per_block,
                    (channels + threads - 1) / threads, 1);
    gp_qwen_causal_conv1d_silu_rows_parallel_kernel<4, rows_per_block>
        <<<grid, threads, 0, (cudaStream_t) stream>>>(
            values, weights, state, out, rows, channels);
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) return (int) err;
    gp_qwen_causal_conv1d_update_state_kernel<4>
        <<<(channels + threads - 1) / threads, threads, 0, (cudaStream_t) stream>>>(
            values, state, rows, channels);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_qwen_normalize_linear_qk_kernel(
        float * __restrict__ q,
        float * __restrict__ k,
        int heads,
        int head_dim,
        float eps) {
    extern __shared__ float s[];
    float * sq = s;
    float * sk = s + blockDim.x;
    const int head = blockIdx.x;
    if (head >= heads) return;
    const int64_t base = (int64_t) head * head_dim;
    float sum_q = 0.0f;
    float sum_k = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float qv = q[base + i];
        const float kv = k[base + i];
        sum_q += qv*qv;
        sum_k += kv*kv;
    }
    sq[threadIdx.x] = sum_q;
    sk[threadIdx.x] = sum_k;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sq[threadIdx.x] += sq[threadIdx.x + stride];
            sk[threadIdx.x] += sk[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float q_scale = rsqrtf(sq[0] + eps) * rsqrtf((float) head_dim);
    const float k_scale = rsqrtf(sk[0] + eps);
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        q[base + i] *= q_scale;
        k[base + i] *= k_scale;
    }
}

GP_CUDA_EXPORT int gp_qwen_normalize_linear_qk(
        float * q,
        float * k,
        int heads,
        int head_dim,
        float eps,
        void * stream) {
    if (heads <= 0 || head_dim <= 0) return (int) cudaErrorInvalidValue;
    gp_qwen_normalize_linear_qk_kernel<<<heads, 256, 2*256*sizeof(float), (cudaStream_t) stream>>>(
        q, k, heads, head_dim, eps);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_qwen_normalize_linear_qk_rows_kernel(
        float * __restrict__ q,
        float * __restrict__ k,
        int rows,
        int heads,
        int head_dim,
        int q_stride,
        int k_stride,
        float eps) {
    extern __shared__ float s[];
    float * sq = s;
    float * sk = s + blockDim.x;
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    if (row >= rows || head >= heads) return;
    const int64_t q_base = (int64_t) row * q_stride + (int64_t) head * head_dim;
    const int64_t k_base = (int64_t) row * k_stride + (int64_t) head * head_dim;
    float sum_q = 0.0f;
    float sum_k = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float qv = q[q_base + i];
        const float kv = k[k_base + i];
        sum_q += qv*qv;
        sum_k += kv*kv;
    }
    sq[threadIdx.x] = sum_q;
    sk[threadIdx.x] = sum_k;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sq[threadIdx.x] += sq[threadIdx.x + stride];
            sk[threadIdx.x] += sk[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float q_scale = rsqrtf(sq[0] + eps) * rsqrtf((float) head_dim);
    const float k_scale = rsqrtf(sk[0] + eps);
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        q[q_base + i] *= q_scale;
        k[k_base + i] *= k_scale;
    }
}

GP_CUDA_EXPORT int gp_qwen_normalize_linear_qk_rows(
        float * q,
        float * k,
        int rows,
        int heads,
        int head_dim,
        int q_stride,
        int k_stride,
        float eps,
        void * stream) {
    if (rows <= 0 || heads <= 0 || head_dim <= 0 ||
        q_stride < heads * head_dim || k_stride < heads * head_dim) {
        return (int) cudaErrorInvalidValue;
    }
    const int block = 128;
    gp_qwen_normalize_linear_qk_rows_kernel<<<dim3(rows, heads, 1), block,
        2*block*sizeof(float), (cudaStream_t) stream>>>(
        q, k, rows, heads, head_dim, q_stride, k_stride, eps);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_qwen_deinterleave_q_gate_kernel(
        const float * __restrict__ packed,
        float * __restrict__ q_out,
        float * __restrict__ gate_out,
        int rows,
        int heads,
        int head_dim,
        int packed_stride,
        int output_stride) {
    const int per_row = heads * head_dim;
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= rows * per_row) return;
    const int row = idx / per_row;
    const int local = idx - row * per_row;
    const int head = local / head_dim;
    const int lane = local - head * head_dim;
    const int64_t src = (int64_t) row * packed_stride + (int64_t) head * head_dim * 2 + lane;
    const int64_t dst = (int64_t) row * output_stride + local;
    q_out[dst] = packed[src];
    gate_out[dst] = packed[src + head_dim];
}

GP_CUDA_EXPORT int gp_qwen_deinterleave_q_gate(
        const float * packed,
        float * q_out,
        float * gate_out,
        int rows,
        int heads,
        int head_dim,
        int packed_stride,
        int output_stride,
        void * stream) {
    if (rows <= 0 || heads <= 0 || head_dim <= 0 ||
        packed_stride < heads * head_dim * 2 || output_stride < heads * head_dim) {
        return (int) cudaErrorInvalidValue;
    }
    const int total = rows * heads * head_dim;
    gp_qwen_deinterleave_q_gate_kernel<<<(total + 255)/256, 256, 0, (cudaStream_t) stream>>>(
        packed, q_out, gate_out, rows, heads, head_dim, packed_stride, output_stride);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_qwen_swiglu_kernel(
        const float * __restrict__ gate,
        const float * __restrict__ up,
        float * __restrict__ dst,
        int total) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= total) return;
    const float x = gate[i];
    dst[i] = (x / (1.0f + expf(-x))) * up[i];
}

GP_CUDA_EXPORT int gp_qwen_swiglu(
        const float * gate,
        const float * up,
        float * dst,
        int total,
        void * stream) {
    if (total <= 0) return (int) cudaErrorInvalidValue;
    gp_qwen_swiglu_kernel<<<(total + 255)/256, 256, 0, (cudaStream_t) stream>>>(
        gate, up, dst, total);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_qwen_apply_silu_gate_kernel(
        float * __restrict__ values,
        const float * __restrict__ gate,
        int total) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= total) return;
    const float x = gate[i];
    values[i] *= x / (1.0f + expf(-x));
}

GP_CUDA_EXPORT int gp_qwen_apply_silu_gate(
        float * values,
        const float * gate,
        int total,
        void * stream) {
    if (total <= 0) return (int) cudaErrorInvalidValue;
    gp_qwen_apply_silu_gate_kernel<<<(total + 255)/256, 256, 0, (cudaStream_t) stream>>>(
        values, gate, total);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_qwen_apply_sigmoid_gate_kernel(
        float * __restrict__ values,
        const float * __restrict__ gate,
        int total) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= total) return;
    values[i] *= 1.0f / (1.0f + expf(-gate[i]));
}

GP_CUDA_EXPORT int gp_qwen_apply_sigmoid_gate(
        float * values,
        const float * gate,
        int total,
        void * stream) {
    if (total <= 0) return (int) cudaErrorInvalidValue;
    gp_qwen_apply_sigmoid_gate_kernel<<<(total + 255)/256, 256, 0, (cudaStream_t) stream>>>(
        values, gate, total);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_qwen_add_kernel(
        const float * __restrict__ lhs,
        const float * __restrict__ rhs,
        float * __restrict__ dst,
        int total) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= total) return;
    dst[i] = lhs[i] + rhs[i];
}

GP_CUDA_EXPORT int gp_qwen_add(
        const float * lhs,
        const float * rhs,
        float * dst,
        int total,
        void * stream) {
    if (total <= 0) return (int) cudaErrorInvalidValue;
    gp_qwen_add_kernel<<<(total + 255)/256, 256, 0, (cudaStream_t) stream>>>(
        lhs, rhs, dst, total);
    return (int) cudaPeekAtLastError();
}

static __device__ __forceinline__ bool gp_qwen_better_logit(float lhs, uint32_t lhs_idx, float rhs, uint32_t rhs_idx) {
    return lhs > rhs || (lhs == rhs && lhs_idx < rhs_idx);
}

__global__ void gp_qwen_argmax_first_kernel(
        const float * __restrict__ logits,
        float * __restrict__ block_values,
        uint32_t * __restrict__ block_indices,
        int total) {
    __shared__ float s_val[256];
    __shared__ uint32_t s_idx[256];
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    float best = -INFINITY;
    uint32_t best_idx = 0xffffffffu;
    if (idx < total) {
        const float v = logits[idx];
        if (!isnan(v)) {
            best = v;
            best_idx = (uint32_t) idx;
        }
    }
    s_val[tid] = best;
    s_idx[tid] = best_idx;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other = s_val[tid + stride];
            const uint32_t other_idx = s_idx[tid + stride];
            if (gp_qwen_better_logit(other, other_idx, s_val[tid], s_idx[tid])) {
                s_val[tid] = other;
                s_idx[tid] = other_idx;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_values[blockIdx.x] = s_val[0];
        block_indices[blockIdx.x] = s_idx[0];
    }
}

__global__ void gp_qwen_argmax_final_kernel(
        const float * __restrict__ block_values,
        const uint32_t * __restrict__ block_indices,
        uint32_t * __restrict__ token_out,
        int blocks) {
    __shared__ float s_val[1024];
    __shared__ uint32_t s_idx[1024];
    const int tid = threadIdx.x;
    float best = -INFINITY;
    uint32_t best_idx = 0xffffffffu;
    for (int i = tid; i < blocks; i += blockDim.x) {
        const float v = block_values[i];
        const uint32_t idx = block_indices[i];
        if (gp_qwen_better_logit(v, idx, best, best_idx)) {
            best = v;
            best_idx = idx;
        }
    }
    s_val[tid] = best;
    s_idx[tid] = best_idx;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other = s_val[tid + stride];
            const uint32_t other_idx = s_idx[tid + stride];
            if (gp_qwen_better_logit(other, other_idx, s_val[tid], s_idx[tid])) {
                s_val[tid] = other;
                s_idx[tid] = other_idx;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        token_out[0] = s_idx[0];
    }
}

GP_CUDA_EXPORT int gp_qwen_argmax(
        const float * logits,
        uint32_t * token_out,
        int total,
        float * block_values,
        uint32_t * block_indices,
        int block_count,
        void * stream) {
    if (total <= 0 || block_count <= 0) return (int) cudaErrorInvalidValue;
    const int expected_blocks = (total + 255) / 256;
    if (block_count < expected_blocks) return (int) cudaErrorInvalidValue;
    gp_qwen_argmax_first_kernel<<<expected_blocks, 256, 0, (cudaStream_t) stream>>>(
        logits, block_values, block_indices, total);
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) return (int) err;
    gp_qwen_argmax_final_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(
        block_values, block_indices, token_out, expected_blocks);
    return (int) cudaPeekAtLastError();
}

static __device__ __forceinline__ float gp_qwen_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void gp_qwen_deltanet_decode_kernel(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ beta,
        const float * __restrict__ alpha,
        const float * __restrict__ a_log,
        const float * __restrict__ dt_bias,
        float * __restrict__ state,
        float * __restrict__ out,
        int heads,
        int head_dim) {
    const int head = blockIdx.x;
    const int value_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (head >= heads || value_idx >= head_dim) return;

    const int64_t head_base = (int64_t) head * head_dim;
    const int64_t row_base = ((int64_t) head * head_dim + value_idx) * head_dim;
    const float beta_h = 1.0f / (1.0f + expf(-beta[head]));
    const float x = alpha[head] + dt_bias[head];
    const float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));
    float decay = expf(-expf(a_log[head]) * sp);
    decay = fminf(fmaxf(decay, 0.0f), 1.0f);

    float prior = 0.0f;
    for (int key_idx = threadIdx.x; key_idx < head_dim; key_idx += blockDim.x) {
        prior += state[row_base + key_idx] * k[head_base + key_idx];
    }
    prior = gp_qwen_warp_sum(prior);
    const float prior_sum = __shfl_sync(0xffffffffu, prior, 0);
    const float delta = (v[head_base + value_idx] - decay * prior_sum) * beta_h;

    float attn = 0.0f;
    for (int key_idx = threadIdx.x; key_idx < head_dim; key_idx += blockDim.x) {
        const int64_t idx = row_base + key_idx;
        const float updated = decay * state[idx] + k[head_base + key_idx] * delta;
        state[idx] = updated;
        attn += updated * q[head_base + key_idx];
    }
    attn = gp_qwen_warp_sum(attn);
    if (threadIdx.x == 0) {
        out[head_base + value_idx] = attn;
    }
}

GP_CUDA_EXPORT int gp_qwen_deltanet_decode(
        const float * q,
        const float * k,
        const float * v,
        const float * beta,
        const float * alpha,
        const float * a_log,
        const float * dt_bias,
        float * state,
        float * out,
        int heads,
        int head_dim,
        void * stream) {
    if (heads <= 0 || head_dim <= 0) return (int) cudaErrorInvalidValue;
    constexpr int warps_per_block = 4;
    const dim3 grid(heads, (head_dim + warps_per_block - 1) / warps_per_block, 1);
    const dim3 block(32, warps_per_block, 1);
    gp_qwen_deltanet_decode_kernel<<<grid, block, 0, (cudaStream_t) stream>>>(
        q, k, v, beta, alpha, a_log, dt_bias, state, out, heads, head_dim);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_qwen_deltanet_decode_rows_kernel(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ beta,
        const float * __restrict__ decay,
        float * __restrict__ state,
        float * __restrict__ out,
        int rows,
        int heads,
        int head_dim,
        int q_stride,
        int k_stride,
        int v_stride,
        int beta_stride,
        int decay_stride,
        int out_stride) {
    const int head = blockIdx.x;
    const int value_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (head >= heads || value_idx >= head_dim) return;

    const int64_t head_base = (int64_t) head * head_dim;
    const int64_t state_base = ((int64_t) head * head_dim + value_idx) * head_dim;
    for (int row = 0; row < rows; ++row) {
        const int64_t q_base = (int64_t) row * q_stride + head_base;
        const int64_t k_base = (int64_t) row * k_stride + head_base;
        const int64_t v_base = (int64_t) row * v_stride + head_base;
        const float beta_h = beta[(int64_t) row * beta_stride + head];
        const float decay_h = decay[(int64_t) row * decay_stride + head];

        if (head_dim == 128 && blockDim.x == 32) {
            float state_v[4];
            float key_v[4];
            float query_v[4];
            float prior = 0.0f;
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                const int key_idx = threadIdx.x + item * 32;
                state_v[item] = state[state_base + key_idx];
                key_v[item] = k[k_base + key_idx];
                query_v[item] = q[q_base + key_idx];
                prior += state_v[item] * key_v[item];
            }
            prior = gp_qwen_warp_sum(prior);
            const float prior_sum = __shfl_sync(0xffffffffu, prior, 0);
            const float delta = (v[v_base + value_idx] - decay_h * prior_sum) * beta_h;

            float attn = 0.0f;
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                const int key_idx = threadIdx.x + item * 32;
                const float updated = decay_h * state_v[item] + key_v[item] * delta;
                state[state_base + key_idx] = updated;
                attn += updated * query_v[item];
            }
            attn = gp_qwen_warp_sum(attn);
            if (threadIdx.x == 0) {
                out[(int64_t) row * out_stride + head_base + value_idx] = attn;
            }
            continue;
        }

        float prior = 0.0f;
        for (int key_idx = threadIdx.x; key_idx < head_dim; key_idx += blockDim.x) {
            prior += state[state_base + key_idx] * k[k_base + key_idx];
        }
        prior = gp_qwen_warp_sum(prior);
        const float prior_sum = __shfl_sync(0xffffffffu, prior, 0);
        const float delta = (v[v_base + value_idx] - decay_h * prior_sum) * beta_h;

        float attn = 0.0f;
        for (int key_idx = threadIdx.x; key_idx < head_dim; key_idx += blockDim.x) {
            const int64_t idx = state_base + key_idx;
            const float updated = decay_h * state[idx] + k[k_base + key_idx] * delta;
            state[idx] = updated;
            attn += updated * q[q_base + key_idx];
        }
        attn = gp_qwen_warp_sum(attn);
        if (threadIdx.x == 0) {
            out[(int64_t) row * out_stride + head_base + value_idx] = attn;
        }
    }
}

template <int value_rows>
__global__ void gp_qwen_deltanet_rows_h16d128_vrows_kernel(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ beta,
        const float * __restrict__ decay,
        float * __restrict__ state,
        float * __restrict__ out,
        int rows) {
    constexpr int heads = 16;
    constexpr int head_dim = 128;
    constexpr int qkv_stride = heads * head_dim * 3;
    constexpr int out_stride = heads * head_dim;
    const int head = blockIdx.x;
    const int value_base =
        (blockIdx.y * blockDim.y + threadIdx.y) * value_rows;
    if (value_base >= head_dim) return;
    const int lane = threadIdx.x;
    const int head_base = head * head_dim;
    float state_values[value_rows][4];
#pragma unroll
    for (int value_row = 0; value_row < value_rows; ++value_row) {
        const int state_base =
            (head * head_dim + value_base + value_row) * head_dim;
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            state_values[value_row][item] =
                state[state_base + lane + item * 32];
        }
    }
    for (int row = 0; row < rows; ++row) {
        const int q_base = row * qkv_stride + head_base;
        const float beta_value = beta[row * heads + head];
        const float decay_value = decay[row * heads + head];
        float key_values[4];
        float query_values[4];
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int idx = lane + item * 32;
            key_values[item] = k[q_base + idx];
            query_values[item] = q[q_base + idx];
        }
#pragma unroll
        for (int value_row = 0; value_row < value_rows; ++value_row) {
            float prior = 0.0f;
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                prior += state_values[value_row][item] * key_values[item];
            }
            prior = gp_qwen_warp_sum(prior);
            const float prior_sum = __shfl_sync(0xffffffffu, prior, 0);
            const int value_idx = value_base + value_row;
            const float delta =
                (v[q_base + value_idx] - decay_value * prior_sum) * beta_value;
            float attention = 0.0f;
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                state_values[value_row][item] =
                    decay_value * state_values[value_row][item] +
                    key_values[item] * delta;
                attention += state_values[value_row][item] * query_values[item];
            }
            attention = gp_qwen_warp_sum(attention);
            if (lane == 0) {
                out[row * out_stride + head_base + value_idx] = attention;
            }
        }
    }
#pragma unroll
    for (int value_row = 0; value_row < value_rows; ++value_row) {
        const int state_base =
            (head * head_dim + value_base + value_row) * head_dim;
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            state[state_base + lane + item * 32] =
                state_values[value_row][item];
        }
    }
}

__global__ void gp_qwen_prepare_deltanet_rows_kernel(
        float * beta,
        float * decay,
        const float * __restrict__ a_log,
        const float * __restrict__ dt_bias,
        int rows,
        int heads,
        int beta_stride,
        int decay_stride) {
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= rows * heads) return;
    const int row = idx / heads;
    const int head = idx - row * heads;
    const int64_t beta_idx = (int64_t) row * beta_stride + head;
    const int64_t decay_idx = (int64_t) row * decay_stride + head;
    beta[beta_idx] = 1.0f / (1.0f + expf(-beta[beta_idx]));
    const float x = decay[decay_idx] + dt_bias[head];
    const float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));
    float value = expf(-expf(a_log[head]) * sp);
    decay[decay_idx] = fminf(fmaxf(value, 0.0f), 1.0f);
}

GP_CUDA_EXPORT int gp_qwen_deltanet_decode_rows(
        const float * q,
        const float * k,
        const float * v,
        float * beta,
        float * alpha,
        const float * a_log,
        const float * dt_bias,
        float * state,
        float * out,
        int rows,
        int heads,
        int head_dim,
        int q_stride,
        int k_stride,
        int v_stride,
        int beta_stride,
        int alpha_stride,
        int out_stride,
        void * stream) {
    if (rows <= 0 || heads <= 0 || head_dim <= 0 ||
        q_stride < heads*head_dim || k_stride < heads*head_dim ||
        v_stride < heads*head_dim || beta_stride < heads || alpha_stride < heads ||
        out_stride < heads*head_dim) {
        return (int) cudaErrorInvalidValue;
    }
    const int prepare_total = rows * heads;
    gp_qwen_prepare_deltanet_rows_kernel<<<(prepare_total + 255)/256, 256, 0,
        (cudaStream_t) stream>>>(
        beta, alpha, a_log, dt_bias, rows, heads, beta_stride, alpha_stride);
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) return (int) err;
    constexpr int warps_per_block = 8;
    const dim3 block(32, warps_per_block, 1);
    if (heads == 16 && head_dim == 128 &&
        q_stride == 6144 && k_stride == 6144 && v_stride == 6144 &&
        beta_stride == 16 && alpha_stride == 16 && out_stride == 2048) {
        const dim3 grid(heads, head_dim / (warps_per_block * 2), 1);
        gp_qwen_deltanet_rows_h16d128_vrows_kernel<2><<<grid, block, 0,
            (cudaStream_t)stream>>>(q, k, v, beta, alpha, state, out, rows);
    } else {
        const dim3 grid(heads, (head_dim + warps_per_block - 1) / warps_per_block, 1);
        gp_qwen_deltanet_decode_rows_kernel<<<grid, block, 0,
            (cudaStream_t) stream>>>(
            q, k, v, beta, alpha, state, out, rows, heads, head_dim,
            q_stride, k_stride, v_stride, beta_stride, alpha_stride, out_stride);
    }
    return (int) cudaPeekAtLastError();
}

__global__ void gp_qwen_rope_decode_kernel(
        float * __restrict__ values,
        int heads,
        int head_dim,
        int rope_dim,
        int position,
        const int * __restrict__ position_ptr,
        float base_freq) {
    if (position_ptr != nullptr) position = *position_ptr;
    const int half = rope_dim / 2;
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = heads * half;
    if (idx >= total) return;
    const int i = idx % half;
    const int head = idx / half;
    const int64_t base = (int64_t) head * head_dim;
    const float x1 = values[base + i];
    const float x2 = values[base + half + i];
    const float inv = powf(base_freq, -2.0f * (float)i / (float)rope_dim);
    const float theta = (float) position * inv;
    float s;
    float c;
    sincosf(theta, &s, &c);
    values[base + i] = x1 * c - x2 * s;
    values[base + half + i] = x1 * s + x2 * c;
}

GP_CUDA_EXPORT int gp_qwen_rope_decode(
        float * values,
        int heads,
        int head_dim,
        int rope_dim,
        int position,
        float base_freq,
        void * stream) {
    if (heads <= 0 || head_dim <= 0 || rope_dim <= 0 || rope_dim > head_dim || (rope_dim & 1) != 0) {
        return (int) cudaErrorInvalidValue;
    }
    const int total = heads * (rope_dim / 2);
    gp_qwen_rope_decode_kernel<<<(total + 255)/256, 256, 0, (cudaStream_t) stream>>>(
        values, heads, head_dim, rope_dim, position, nullptr, base_freq);
    return (int) cudaPeekAtLastError();
}

GP_CUDA_EXPORT int gp_qwen_rope_decode_position(
        float * values,
        int heads,
        int head_dim,
        int rope_dim,
        const int * position,
        float base_freq,
        void * stream) {
    if (heads <= 0 || head_dim <= 0 || rope_dim <= 0 || rope_dim > head_dim ||
        (rope_dim & 1) != 0 || position == nullptr) {
        return (int)cudaErrorInvalidValue;
    }
    const int total = heads * (rope_dim / 2);
    gp_qwen_rope_decode_kernel<<<(total + 255)/256, 256, 0, (cudaStream_t)stream>>>(
        values, heads, head_dim, rope_dim, 0, position, base_freq);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_qwen_rope_rows_kernel(
        float * __restrict__ values,
        int rows,
        int heads,
        int head_dim,
        int rope_dim,
        int position,
        int row_stride,
        float base_freq) {
    const int half = rope_dim / 2;
    const int per_row = heads * half;
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= rows * per_row) return;
    const int row = idx / per_row;
    const int local = idx - row * per_row;
    const int i = local % half;
    const int head = local / half;
    const int64_t base = (int64_t) row * row_stride + (int64_t) head * head_dim;
    const float x1 = values[base + i];
    const float x2 = values[base + half + i];
    const float inv = powf(base_freq, -2.0f * (float)i / (float)rope_dim);
    const float theta = (float)(position + row) * inv;
    float s;
    float c;
    sincosf(theta, &s, &c);
    values[base + i] = x1 * c - x2 * s;
    values[base + half + i] = x1 * s + x2 * c;
}

GP_CUDA_EXPORT int gp_qwen_rope_rows(
        float * values,
        int rows,
        int heads,
        int head_dim,
        int rope_dim,
        int position,
        int row_stride,
        float base_freq,
        void * stream) {
    if (rows <= 0 || heads <= 0 || head_dim <= 0 || rope_dim <= 0 ||
        rope_dim > head_dim || (rope_dim & 1) != 0 || position < 0 ||
        row_stride < heads * head_dim) {
        return (int) cudaErrorInvalidValue;
    }
    const int total = rows * heads * (rope_dim / 2);
    gp_qwen_rope_rows_kernel<<<(total + 255)/256, 256, 0, (cudaStream_t) stream>>>(
        values, rows, heads, head_dim, rope_dim, position, row_stride, base_freq);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_qwen_cache_write_kernel(
        const float * __restrict__ src,
        float * __restrict__ cache,
        int position,
        const int * __restrict__ position_ptr,
        int heads,
        int head_dim,
        int max_context) {
    if (position_ptr != nullptr) position = *position_ptr;
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = heads * head_dim;
    if (position < 0 || position >= max_context || idx >= total) return;
    cache[((int64_t) position * heads * head_dim) + idx] = src[idx];
}

GP_CUDA_EXPORT int gp_qwen_cache_write(
        const float * src,
        float * cache,
        int position,
        int heads,
        int head_dim,
        int max_context,
        void * stream) {
    if (position < 0 || position >= max_context || heads <= 0 || head_dim <= 0) {
        return (int) cudaErrorInvalidValue;
    }
    const int total = heads * head_dim;
    gp_qwen_cache_write_kernel<<<(total + 255)/256, 256, 0, (cudaStream_t) stream>>>(
        src, cache, position, nullptr, heads, head_dim, max_context);
    return (int) cudaPeekAtLastError();
}

GP_CUDA_EXPORT int gp_qwen_cache_write_position(
        const float * src,
        float * cache,
        const int * position,
        int heads,
        int head_dim,
        int max_context,
        void * stream) {
    if (position == nullptr || heads <= 0 || head_dim <= 0 || max_context <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    const int total = heads * head_dim;
    gp_qwen_cache_write_kernel<<<(total + 255)/256, 256, 0, (cudaStream_t)stream>>>(
        src, cache, 0, position, heads, head_dim, max_context);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_qwen_cache_write_rows_kernel(
        const float * __restrict__ src,
        float * __restrict__ cache,
        int rows,
        int position,
        int heads,
        int head_dim,
        int src_stride) {
    const int width = heads * head_dim;
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= rows * width) return;
    const int row = idx / width;
    const int local = idx - row * width;
    cache[((int64_t)(position + row) * width) + local] = src[(int64_t)row * src_stride + local];
}

GP_CUDA_EXPORT int gp_qwen_cache_write_rows(
        const float * src,
        float * cache,
        int rows,
        int position,
        int heads,
        int head_dim,
        int max_context,
        int src_stride,
        void * stream) {
    if (rows <= 0 || position < 0 || position + rows > max_context ||
        heads <= 0 || head_dim <= 0 || src_stride < heads * head_dim) {
        return (int) cudaErrorInvalidValue;
    }
    const int total = rows * heads * head_dim;
    gp_qwen_cache_write_rows_kernel<<<(total + 255)/256, 256, 0, (cudaStream_t) stream>>>(
        src, cache, rows, position, heads, head_dim, src_stride);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_qwen_attention_scores_decode_kernel(
        const float * __restrict__ q,
        const float * __restrict__ k_cache,
        float * __restrict__ scores,
        int position,
        const int * __restrict__ position_ptr,
        int q_heads,
        int kv_heads,
        int head_dim,
        int max_context,
        float scale) {
    if (position_ptr != nullptr) position = *position_ptr;
    extern __shared__ float s[];
    const int pos = blockIdx.x;
    const int q_head = blockIdx.y;
    if (position < 0 || position >= max_context || pos > position || q_head >= q_heads) return;
    const int gqa = q_heads / kv_heads;
    const int kv_head = q_head / gqa;
    const int64_t q_base = (int64_t) q_head * head_dim;
    const int64_t k_base = ((int64_t) pos * kv_heads + kv_head) * head_dim;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        acc += q[q_base + i] * k_cache[k_base + i];
    }
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        scores[(int64_t) q_head * max_context + pos] = s[0] * scale;
    }
}

GP_CUDA_EXPORT int gp_qwen_attention_scores_decode(
        const float * q,
        const float * k_cache,
        float * scores,
        int position,
        int q_heads,
        int kv_heads,
        int head_dim,
        int max_context,
        float scale,
        void * stream) {
    if (position < 0 || position >= max_context || q_heads <= 0 || kv_heads <= 0 ||
        head_dim <= 0 || q_heads % kv_heads != 0) {
        return (int) cudaErrorInvalidValue;
    }
    gp_qwen_attention_scores_decode_kernel<<<dim3(position + 1, q_heads, 1), 256, 256*sizeof(float), (cudaStream_t) stream>>>(
        q, k_cache, scores, position, nullptr, q_heads, kv_heads, head_dim, max_context, scale);
    return (int) cudaPeekAtLastError();
}

GP_CUDA_EXPORT int gp_qwen_attention_scores_decode_position(
        const float * q,
        const float * k_cache,
        float * scores,
        const int * position,
        int q_heads,
        int kv_heads,
        int head_dim,
        int max_context,
        float scale,
        void * stream) {
    if (position == nullptr || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
        max_context <= 0 || q_heads % kv_heads != 0) {
        return (int)cudaErrorInvalidValue;
    }
    gp_qwen_attention_scores_decode_kernel<<<dim3(max_context, q_heads, 1), 256,
        256*sizeof(float), (cudaStream_t)stream>>>(
        q, k_cache, scores, 0, position, q_heads, kv_heads, head_dim, max_context, scale);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_qwen_softmax_decode_kernel(
        float * __restrict__ scores,
        int position,
        const int * __restrict__ position_ptr,
        int heads,
        int max_context) {
    if (position_ptr != nullptr) position = *position_ptr;
    extern __shared__ float s[];
    const int head = blockIdx.x;
    if (position < 0 || position >= max_context || head >= heads) return;
    float local_max = -INFINITY;
    const int64_t base = (int64_t) head * max_context;
    for (int pos = threadIdx.x; pos <= position; pos += blockDim.x) {
        local_max = fmaxf(local_max, scores[base + pos]);
    }
    s[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + stride]);
        __syncthreads();
    }
    const float max_v = s[0];
    float local_sum = 0.0f;
    for (int pos = threadIdx.x; pos <= position; pos += blockDim.x) {
        const float v = expf(scores[base + pos] - max_v);
        scores[base + pos] = v;
        local_sum += v;
    }
    s[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = 1.0f / fmaxf(s[0], 1.0e-20f);
    for (int pos = threadIdx.x; pos <= position; pos += blockDim.x) {
        scores[base + pos] *= inv;
    }
}

GP_CUDA_EXPORT int gp_qwen_softmax_decode(
        float * scores,
        int position,
        int heads,
        int max_context,
        void * stream) {
    if (position < 0 || position >= max_context || heads <= 0) return (int) cudaErrorInvalidValue;
    gp_qwen_softmax_decode_kernel<<<heads, 256, 256*sizeof(float), (cudaStream_t) stream>>>(
        scores, position, nullptr, heads, max_context);
    return (int) cudaPeekAtLastError();
}

GP_CUDA_EXPORT int gp_qwen_softmax_decode_position(
        float * scores,
        const int * position,
        int heads,
        int max_context,
        void * stream) {
    if (position == nullptr || heads <= 0 || max_context <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    gp_qwen_softmax_decode_kernel<<<heads, 256, 256*sizeof(float), (cudaStream_t)stream>>>(
        scores, 0, position, heads, max_context);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_qwen_attention_values_decode_kernel(
        const float * __restrict__ scores,
        const float * __restrict__ v_cache,
        float * __restrict__ out,
        int position,
        const int * __restrict__ position_ptr,
        int q_heads,
        int kv_heads,
        int value_dim,
        int max_context) {
    if (position_ptr != nullptr) position = *position_ptr;
    extern __shared__ float s[];
    const int q_head = blockIdx.x;
    const int value_idx = blockIdx.y;
    if (position < 0 || position >= max_context || q_head >= q_heads || value_idx >= value_dim) return;
    const int gqa = q_heads / kv_heads;
    const int kv_head = q_head / gqa;
    float acc = 0.0f;
    const int64_t score_base = (int64_t) q_head * max_context;
    for (int pos = threadIdx.x; pos <= position; pos += blockDim.x) {
        const int64_t v_base = ((int64_t) pos * kv_heads + kv_head) * value_dim;
        acc += scores[score_base + pos] * v_cache[v_base + value_idx];
    }
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[(int64_t) q_head * value_dim + value_idx] = s[0];
    }
}

GP_CUDA_EXPORT int gp_qwen_attention_values_decode(
        const float * scores,
        const float * v_cache,
        float * out,
        int position,
        int q_heads,
        int kv_heads,
        int value_dim,
        int max_context,
        void * stream) {
    if (position < 0 || position >= max_context || q_heads <= 0 || kv_heads <= 0 ||
        value_dim <= 0 || q_heads % kv_heads != 0) {
        return (int) cudaErrorInvalidValue;
    }
    gp_qwen_attention_values_decode_kernel<<<dim3(q_heads, value_dim, 1), 256, 256*sizeof(float), (cudaStream_t) stream>>>(
        scores, v_cache, out, position, nullptr, q_heads, kv_heads, value_dim, max_context);
    return (int) cudaPeekAtLastError();
}

GP_CUDA_EXPORT int gp_qwen_attention_values_decode_position(
        const float * scores,
        const float * v_cache,
        float * out,
        const int * position,
        int q_heads,
        int kv_heads,
        int value_dim,
        int max_context,
        void * stream) {
    if (position == nullptr || q_heads <= 0 || kv_heads <= 0 || value_dim <= 0 ||
        max_context <= 0 || q_heads % kv_heads != 0) {
        return (int)cudaErrorInvalidValue;
    }
    gp_qwen_attention_values_decode_kernel<<<dim3(q_heads, value_dim, 1), 256,
        256*sizeof(float), (cudaStream_t)stream>>>(
        scores, v_cache, out, 0, position, q_heads, kv_heads, value_dim, max_context);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_qwen_increment_position_kernel(int * position) {
    if (threadIdx.x == 0) ++*position;
}

GP_CUDA_EXPORT int gp_qwen_increment_position(int * position, void * stream) {
    if (position == nullptr) return (int)cudaErrorInvalidValue;
    gp_qwen_increment_position_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(position);
    return (int)cudaPeekAtLastError();
}

__global__ void gp_qwen_attention_rows_fused_kernel(
        const float * __restrict__ q,
        const float * __restrict__ k_cache,
        const float * __restrict__ v_cache,
        float * __restrict__ out,
        int rows,
        int position,
        int q_heads,
        int kv_heads,
        int head_dim,
        int value_dim,
        int q_stride,
        float scale) {
    extern __shared__ float shared[];
    const int row = blockIdx.x;
    const int q_head = blockIdx.y;
    const int context = position + row + 1;
    float * scores = shared;
    float * reduce = scores + context;
    if (row >= rows || q_head >= q_heads) return;
    const int gqa = q_heads / kv_heads;
    const int kv_head = q_head / gqa;
    const int64_t q_base = (int64_t) row * q_stride + (int64_t) q_head * head_dim;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps = blockDim.x >> 5;
    float q_values[4];
    if (head_dim == 128) {
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            q_values[item] = q[q_base + lane + item * 32];
        }
    }
    float local_max = -INFINITY;
    for (int pos = warp; pos < context; pos += warps) {
        const int64_t k_base = ((int64_t) pos * kv_heads + kv_head) * head_dim;
        float acc = 0.0f;
        if (head_dim == 128) {
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                const int i = lane + item * 32;
                acc += q_values[item] * k_cache[k_base + i];
            }
        } else {
            for (int i = lane; i < head_dim; i += 32) {
                acc += q[q_base + i] * k_cache[k_base + i];
            }
        }
        acc = gp_qwen_warp_sum(acc);
        if (lane == 0) {
            const float score = acc * scale;
            scores[pos] = score;
            local_max = fmaxf(local_max, score);
        }
    }
    reduce[threadIdx.x] = lane == 0 ? local_max : -INFINITY;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        __syncthreads();
    }
    const float max_v = reduce[0];

    float local_sum = 0.0f;
    for (int pos = threadIdx.x; pos < context; pos += blockDim.x) {
        const float value = expf(scores[pos] - max_v);
        scores[pos] = value;
        local_sum += value;
    }
    reduce[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv_sum = 1.0f / fmaxf(reduce[0], 1.0e-20f);
    for (int pos = threadIdx.x; pos < context; pos += blockDim.x) {
        scores[pos] *= inv_sum;
    }
    __syncthreads();

    for (int value_idx = threadIdx.x; value_idx < value_dim; value_idx += blockDim.x) {
        float acc = 0.0f;
        for (int pos = 0; pos < context; ++pos) {
            const int64_t v_base = ((int64_t) pos * kv_heads + kv_head) * value_dim;
            acc += scores[pos] * v_cache[v_base + value_idx];
        }
        out[((int64_t) row * q_heads + q_head) * value_dim + value_idx] = acc;
    }
}

template <int query_tile>
__global__ void gp_qwen_attention_rows_qtile_h256v256_kernel(
        const float * __restrict__ q,
        const float * __restrict__ k_cache,
        const float * __restrict__ v_cache,
        float * __restrict__ out,
        int rows,
        int position,
        int q_heads,
        int kv_heads,
        int q_stride,
        float scale) {
    extern __shared__ float shared_scores[];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * query_tile + warp;
    if (row >= rows) return;

    const int q_head = blockIdx.y;
    const int kv_head = q_head / (q_heads / kv_heads);
    const int full_context = position + rows;
    const int context = position + row + 1;
    float * scores = shared_scores + (int64_t)warp * full_context;
    const int64_t q_base = (int64_t)row * q_stride + (int64_t)q_head * 256;
    float q_values[8];
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        q_values[item] = q[q_base + lane + item * 32];
    }

    float local_max = -INFINITY;
    for (int pos = 0; pos < context; ++pos) {
        const int64_t k_base = ((int64_t)pos * kv_heads + kv_head) * 256;
        float score = 0.0f;
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            score += q_values[item] * k_cache[k_base + lane + item * 32];
        }
        score = gp_qwen_warp_sum(score);
        if (lane == 0) {
            score *= scale;
            scores[pos] = score;
            local_max = fmaxf(local_max, score);
        }
    }
    const float max_value = __shfl_sync(0xffffffffu, local_max, 0);

    float local_sum = 0.0f;
    for (int pos = lane; pos < context; pos += 32) {
        const float value = expf(scores[pos] - max_value);
        scores[pos] = value;
        local_sum += value;
    }
    local_sum = gp_qwen_warp_sum(local_sum);
    const float inv_sum = 1.0f / fmaxf(__shfl_sync(0xffffffffu, local_sum, 0), 1.0e-20f);
    for (int pos = lane; pos < context; pos += 32) {
        scores[pos] *= inv_sum;
    }
    __syncwarp();

    float values[8];
#pragma unroll
    for (int item = 0; item < 8; ++item) values[item] = 0.0f;
    for (int pos = 0; pos < context; ++pos) {
        const int64_t v_base = ((int64_t)pos * kv_heads + kv_head) * 256;
        const float weight = scores[pos];
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            values[item] += weight * v_cache[v_base + lane + item * 32];
        }
    }
    const int64_t out_base = ((int64_t)row * q_heads + q_head) * 256;
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        out[out_base + lane + item * 32] = values[item];
    }
}

GP_CUDA_EXPORT int gp_qwen_attention_rows_fused(
        const float * q,
        const float * k_cache,
        const float * v_cache,
        float * out,
        int rows,
        int position,
        int q_heads,
        int kv_heads,
        int head_dim,
        int value_dim,
        int max_context,
        int q_stride,
        void * stream) {
    if (rows <= 0 || position < 0 || position + rows > max_context ||
        q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0 ||
        head_dim <= 0 || value_dim <= 0 || q_stride < q_heads * head_dim) {
        return (int) cudaErrorInvalidValue;
    }
    if (head_dim == 256 && value_dim == 256) {
        constexpr int query_tile = 2;
        const int block = query_tile * 32;
        const size_t shared = (size_t)query_tile * (position + rows) * sizeof(float);
        if (shared > 48*1024) {
            int max_shared = 0;
            cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
            if (shared > (size_t)max_shared) return (int)cudaErrorInvalidValue;
            cudaError_t attr = cudaFuncSetAttribute(
                (const void *)gp_qwen_attention_rows_qtile_h256v256_kernel<query_tile>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared);
            if (attr != cudaSuccess) return (int)attr;
        }
        const float scale = 0.0625f;
        gp_qwen_attention_rows_qtile_h256v256_kernel<query_tile><<<
            dim3((rows + query_tile - 1) / query_tile, q_heads, 1), block, shared,
            (cudaStream_t)stream>>>(q, k_cache, v_cache, out, rows, position,
            q_heads, kv_heads, q_stride, scale);
        return (int)cudaPeekAtLastError();
    }

    const int block = 128;
    const size_t shared = (size_t)(position + rows + block + head_dim) * sizeof(float);
    if (shared > 48*1024) {
        int max_shared = 0;
        cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
        if (shared > (size_t)max_shared) return (int)cudaErrorInvalidValue;
        cudaError_t attr = cudaFuncSetAttribute((const void *)gp_qwen_attention_rows_fused_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared);
        if (attr != cudaSuccess) return (int)attr;
    }
    const float scale = rsqrtf((float)head_dim);
    gp_qwen_attention_rows_fused_kernel<<<dim3(rows, q_heads, 1), block, shared,
        (cudaStream_t) stream>>>(q, k_cache, v_cache, out, rows, position, q_heads,
        kv_heads, head_dim, value_dim, q_stride, scale);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_rms_norm_heads_kernel(
        const float * __restrict__ src,
        const float * __restrict__ weight,
        float * __restrict__ dst,
        int batch,
        int seq,
        int heads,
        int head_dim,
        int row_width,
        float eps) {
    extern __shared__ float s[];
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int pos = blockIdx.x;
    float sum = 0.0f;
    const int64_t src_base = ((int64_t) b*seq + pos)*row_width + h*head_dim;
    const int64_t dst_base = ((int64_t) b*heads + h)*seq*head_dim + pos*head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float v = src[src_base + i];
        sum += v*v;
    }
    s[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(s[0] / (float) head_dim + eps);
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        dst[dst_base + i] = src[src_base + i] * inv * weight[i];
    }
}

GP_CUDA_EXPORT int gp_rms_norm_heads(
        const float * src, const float * weight, float * dst,
        int batch, int seq, int heads, int head_dim, int row_width,
        float eps, void * stream) {
    gp_rms_norm_heads_kernel<<<dim3(seq, heads, batch), 256, 256*sizeof(float), (cudaStream_t) stream>>>(
        src, weight, dst, batch, seq, heads, head_dim, row_width, eps);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_split_heads_kernel(
        const float * __restrict__ src,
        float * __restrict__ dst,
        int batch,
        int seq,
        int heads,
        int head_dim,
        int row_width) {
    const int idx = (int)(blockIdx.x*blockDim.x + threadIdx.x);
    const int total = batch*seq*heads*head_dim;
    if (idx >= total) return;
    const int i = idx % head_dim;
    const int h = (idx / head_dim) % heads;
    const int pos = (idx / (head_dim*heads)) % seq;
    const int b = idx / (head_dim*heads*seq);
    dst[((int64_t)b*heads + h)*seq*head_dim + pos*head_dim + i] =
        src[((int64_t)b*seq + pos)*row_width + h*head_dim + i];
}

GP_CUDA_EXPORT int gp_split_heads(
        const float * src, float * dst,
        int batch, int seq, int heads, int head_dim, int row_width,
        void * stream) {
    const int total = batch*seq*heads*head_dim;
    gp_split_heads_kernel<<<(total + 255)/256, 256, 0, (cudaStream_t) stream>>>(
        src, dst, batch, seq, heads, head_dim, row_width);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_rope_neox_kernel(
        const float * __restrict__ src,
        float * __restrict__ dst,
        int total_pairs,
        int seq,
        int head_dim,
        float base_freq) {
    const int idx = (int)(blockIdx.x*blockDim.x + threadIdx.x);
    const int half = head_dim / 2;
    if (idx >= total_pairs) return;
    const int i = idx % half;
    const int pos = (idx / half) % seq;
    const int row = idx / half;
    const int64_t off = (int64_t) row*head_dim;
    const float x1 = src[off + i];
    const float x2 = src[off + half + i];
    const float inv = powf(base_freq, -2.0f*(float)i/(float)head_dim);
    const float f = (float)pos * inv;
    const float c = cosf(f);
    const float s = sinf(f);
    dst[off + i] = x1*c - x2*s;
    dst[off + half + i] = x2*c + x1*s;
}

GP_CUDA_EXPORT int gp_rope_neox(
        const float * src, float * dst,
        int batch, int seq, int heads, int head_dim,
        float base_freq, void * stream) {
    const int total_pairs = batch*heads*seq*(head_dim/2);
    gp_rope_neox_kernel<<<(total_pairs + 255)/256, 256, 0, (cudaStream_t) stream>>>(
        src, dst, total_pairs, seq, head_dim, base_freq);
    return (int) cudaPeekAtLastError();
}

GP_CUDA_EXPORT int gp_attention_scores(
        void * blas,
        const float * q,
        const float * k,
        float * scores,
        int batch,
        int heads,
        int seq,
        int head_dim) {
    cublasHandle_t handle = (cublasHandle_t) blas;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (batch == 1) {
        cublasStatus_t st = cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            seq,
            seq,
            head_dim,
            &alpha,
            k,
            head_dim,
            0,
            q,
            head_dim,
            (long long) seq*head_dim,
            &beta,
            scores,
            seq,
            (long long) seq*seq,
            heads);
        return st == CUBLAS_STATUS_SUCCESS ? 0 : 10000 + (int) st;
    }
    for (int b = 0; b < batch; ++b) {
        const float * kb = k + (int64_t)b*seq*head_dim;
        for (int h = 0; h < heads; ++h) {
            const float * qh = q + ((int64_t)b*heads + h)*seq*head_dim;
            float * sh = scores + ((int64_t)b*heads + h)*seq*seq;
            cublasStatus_t st = cublasSgemm(
                handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                seq,
                seq,
                head_dim,
                &alpha,
                kb,
                head_dim,
                qh,
                head_dim,
                &beta,
                sh,
                seq);
            if (st != CUBLAS_STATUS_SUCCESS) return 10000 + (int) st;
        }
    }
    return 0;
}

__global__ void gp_softmax_mask_kernel(
        float * __restrict__ scores,
        const uint32_t * __restrict__ mask,
        int batch,
        int heads,
        int seq,
        int sliding_window,
        float scale) {
    extern __shared__ float s[];
    const int q = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    float local_max = -INFINITY;
    const int64_t base = ((int64_t)b*heads + h)*seq*seq + (int64_t)q*seq;
    for (int k = threadIdx.x; k < seq; k += blockDim.x) {
        const bool key_visible = mask[(int64_t)b*seq + k] != 0;
        const bool in_window = sliding_window <= 0 || abs(q - k) < sliding_window;
        const float v = (key_visible && in_window) ? scores[base + k] * scale : -1.0e9f;
        scores[base + k] = v;
        local_max = fmaxf(local_max, v);
    }
    s[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + stride]);
        __syncthreads();
    }
    const float max_v = s[0];
    float local_sum = 0.0f;
    for (int k = threadIdx.x; k < seq; k += blockDim.x) {
        const float v = expf(scores[base + k] - max_v);
        scores[base + k] = v;
        local_sum += v;
    }
    s[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = 1.0f / fmaxf(s[0], 1.0e-20f);
    for (int k = threadIdx.x; k < seq; k += blockDim.x) {
        scores[base + k] *= inv;
    }
}

GP_CUDA_EXPORT int gp_softmax_mask(
        float * scores,
        const uint32_t * mask,
        int batch,
        int heads,
        int seq,
        int sliding_window,
        float scale,
        void * stream) {
    gp_softmax_mask_kernel<<<dim3(seq, heads, batch), 256, 256*sizeof(float), (cudaStream_t) stream>>>(
        scores, mask, batch, heads, seq, sliding_window, scale);
    return (int) cudaPeekAtLastError();
}

GP_CUDA_EXPORT int gp_attention_values(
        void * blas,
        const float * scores,
        const float * v,
        float * out,
        int batch,
        int heads,
        int seq,
        int head_dim) {
    cublasHandle_t handle = (cublasHandle_t) blas;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (batch == 1) {
        cublasStatus_t st = cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            head_dim,
            seq,
            seq,
            &alpha,
            v,
            head_dim,
            0,
            scores,
            seq,
            (long long) seq*seq,
            &beta,
            out,
            head_dim,
            (long long) seq*head_dim,
            heads);
        return st == CUBLAS_STATUS_SUCCESS ? 0 : 10000 + (int) st;
    }
    for (int b = 0; b < batch; ++b) {
        const float * vb = v + (int64_t)b*seq*head_dim;
        for (int h = 0; h < heads; ++h) {
            const float * sh = scores + ((int64_t)b*heads + h)*seq*seq;
            float * oh = out + ((int64_t)b*heads + h)*seq*head_dim;
            cublasStatus_t st = cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                head_dim,
                seq,
                seq,
                &alpha,
                vb,
                head_dim,
                sh,
                seq,
                &beta,
                oh,
                head_dim);
            if (st != CUBLAS_STATUS_SUCCESS) return 10000 + (int) st;
        }
    }
    return 0;
}

__global__ void gp_merge_heads_kernel(
        const float * __restrict__ src,
        float * __restrict__ dst,
        int batch,
        int seq,
        int heads,
        int head_dim) {
    const int idx = (int)(blockIdx.x*blockDim.x + threadIdx.x);
    const int total = batch*seq*heads*head_dim;
    if (idx >= total) return;
    const int i = idx % head_dim;
    const int h = (idx / head_dim) % heads;
    const int pos = (idx / (head_dim*heads)) % seq;
    const int b = idx / (head_dim*heads*seq);
    dst[((int64_t)b*seq + pos)*heads*head_dim + h*head_dim + i] =
        src[((int64_t)b*heads + h)*seq*head_dim + pos*head_dim + i];
}

GP_CUDA_EXPORT int gp_merge_heads(
        const float * src, float * dst,
        int batch, int seq, int heads, int head_dim,
        void * stream) {
    const int total = batch*seq*heads*head_dim;
    gp_merge_heads_kernel<<<(total + 255)/256, 256, 0, (cudaStream_t) stream>>>(
        src, dst, batch, seq, heads, head_dim);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_geglu_kernel(
        const float * __restrict__ gate,
        const float * __restrict__ up,
        float * __restrict__ dst,
        int total) {
    const int i = (int)(blockIdx.x*blockDim.x + threadIdx.x);
    if (i >= total) return;
    const float x = gate[i];
    const float x3 = x*x*x;
    const float gelu = 0.5f*x*(1.0f + tanhf(0.7978845608028654f*(x + 0.044715f*x3)));
    dst[i] = gelu * up[i];
}

GP_CUDA_EXPORT int gp_geglu(
        const float * gate,
        const float * up,
        float * dst,
        int total,
        void * stream) {
    gp_geglu_kernel<<<(total + 255)/256, 256, 0, (cudaStream_t) stream>>>(
        gate, up, dst, total);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_mean_pool_kernel(
        const float * __restrict__ hidden,
        const uint32_t * __restrict__ mask,
        float * __restrict__ dst,
        int batch,
        int seq,
        int hidden_dim) {
    const int h = (int)(blockIdx.x*blockDim.x + threadIdx.x);
    const int b = blockIdx.y;
    if (h >= hidden_dim) return;
    float sum = 0.0f;
    int count = 0;
    for (int s = 0; s < seq; ++s) {
        if (mask[(int64_t)b*seq + s] != 0) {
            sum += hidden[((int64_t)b*seq + s)*hidden_dim + h];
            count += 1;
        }
    }
    dst[(int64_t)b*hidden_dim + h] = sum / (float) max(count, 1);
}

GP_CUDA_EXPORT int gp_mean_pool(
        const float * hidden,
        const uint32_t * mask,
        float * dst,
        int batch,
        int seq,
        int hidden_dim,
        void * stream) {
    gp_mean_pool_kernel<<<dim3((hidden_dim + 255)/256, batch, 1), 256, 0, (cudaStream_t) stream>>>(
        hidden, mask, dst, batch, seq, hidden_dim);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_l2_norm_kernel(
        const float * __restrict__ src,
        float * __restrict__ dst,
        int rows,
        int dim) {
    extern __shared__ float s[];
    const int row = blockIdx.x;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = src[(int64_t)row*dim + i];
        sum += v*v;
    }
    s[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(fmaxf(s[0], 1.0e-12f));
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        dst[(int64_t)row*dim + i] = src[(int64_t)row*dim + i] * inv;
    }
}

GP_CUDA_EXPORT int gp_l2_norm(
        const float * src,
        float * dst,
        int rows,
        int dim,
        void * stream) {
    gp_l2_norm_kernel<<<rows, 256, 256*sizeof(float), (cudaStream_t) stream>>>(
        src, dst, rows, dim);
    return (int) cudaPeekAtLastError();
}

static int gp_blck_size(int dtype) {
    switch ((ggml_type) dtype) {
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
            return 32;
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return 256;
        default:
            return 0;
    }
}

template <ggml_type type, int mmq_x>
static cudaError_t gp_configure_mmq_shared(const int nbytes_shared) {
    static const cudaError_t configure_error = [nbytes_shared]() {
        cudaError_t err = cudaFuncSetAttribute(
            (const void *)mul_mat_q<type, mmq_x, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared);
        if (err != cudaSuccess) return err;
        return cudaFuncSetAttribute(
            (const void *)mul_mat_q<type, mmq_x, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared);
    }();
    return configure_error;
}

template <ggml_type type, int mmq_x, bool use_stream_k, bool need_check>
static cudaError_t gp_launch_mmq_typed(
        const char * weights,
        const int * y_q8,
        float * dst,
        float * fixup,
        int64_t ncols_x,
        int64_t stride_row_x,
        int64_t nrows_x,
        int64_t ncols_dst,
        int nsm,
        cudaStream_t stream) {
    constexpr int cc = 860;
    constexpr int warp_size = 32;
    constexpr int nwarps = 8;
    constexpr int mmq_y = 128;
    const int nbytes_shared = (int) mmq_get_nbytes_shared<type>(mmq_x, mmq_y, cc, warp_size, nwarps);
    const cudaError_t configure_error =
        gp_configure_mmq_shared<type, mmq_x>(nbytes_shared);
    if (configure_error != cudaSuccess) return configure_error;

    const dim3 block_dims(warp_size, nwarps, 1);
    const int64_t nrows_dst = nrows_x;
    const int64_t ncols_y = ncols_dst;
    const int64_t ncols_max = ncols_dst;
    const int nty = (int)((nrows_x + mmq_y - 1) / mmq_y);
    const int ntx = (int)((ncols_max + mmq_x - 1) / mmq_x);

    const uint3 blocks_per_ne00 = init_fastdiv_values(
        ncols_x / ggml_cuda_type_traits<type>::qk);
    const uint3 ntx_fast = init_fastdiv_values(ntx);
    const uint3 one = init_fastdiv_values(1);
    const int zero = 0;
    if constexpr (!use_stream_k) {
        const dim3 grid(nty, ntx, 1);
        mul_mat_q<type, mmq_x, need_check><<<grid, block_dims, nbytes_shared, stream>>>(
            weights, y_q8, nullptr, nullptr, dst, nullptr,
            blocks_per_ne00, nrows_x, ncols_dst, stride_row_x, ncols_y, nrows_dst,
            one, one, zero, zero, zero,
            one, one, zero, zero, zero,
            ntx_fast);
        return cudaPeekAtLastError();
    } else {
        const int tile_count = ntx*nty;
        const int grid_x = ncols_dst <= 128 ? nsm : min(tile_count, nsm);
        const dim3 grid(grid_x, 1, 1);
        const bool fixup_needed = tile_count % grid_x != 0;
        float * tmp = fixup_needed ? fixup : nullptr;
        mul_mat_q<type, mmq_x, need_check><<<grid, block_dims, nbytes_shared, stream>>>(
            weights, y_q8, nullptr, nullptr, dst, tmp,
            blocks_per_ne00, nrows_x, ncols_dst, stride_row_x, ncols_y, nrows_dst,
            one, one, zero, zero, zero,
            one, one, zero, zero, zero,
            ntx_fast);
        cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess || !fixup_needed) return err;
        const dim3 fixup_grid(grid_x, mmq_y / warp_size, 1);
        const dim3 fixup_block(warp_size, nwarps / 2, 1);
        mul_mat_q_stream_k_fixup<type, mmq_x, need_check><<<fixup_grid, fixup_block, 0, stream>>>(
            nullptr, nullptr, dst, tmp,
            blocks_per_ne00, nrows_x, ncols_dst, nrows_dst,
            one, zero, one, zero, ntx_fast);
        return cudaPeekAtLastError();
    }
}

template <ggml_type type, int mmq_x>
static cudaError_t gp_launch_mmq_choose_stream(
        const char * weights,
        const int * y_q8,
        float * dst,
        float * fixup,
        int64_t ncols_x,
        int64_t stride_row_x,
        int64_t nrows_x,
        int64_t ncols_dst,
        int nsm,
        bool use_stream_k,
        cudaStream_t stream) {
    constexpr int mmq_y = 128;
    const bool need_check = nrows_x % mmq_y != 0;
    if (use_stream_k) {
        if (need_check) {
            return gp_launch_mmq_typed<type, mmq_x, true, true>(
                weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, stream);
        }
        return gp_launch_mmq_typed<type, mmq_x, true, false>(
            weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, stream);
    }
    if (need_check) {
        return gp_launch_mmq_typed<type, mmq_x, false, true>(
            weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, stream);
    }
    return gp_launch_mmq_typed<type, mmq_x, false, false>(
        weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, stream);
}

template <ggml_type type>
static cudaError_t gp_launch_mmq_type(
        const char * weights,
        const int * y_q8,
        float * dst,
        float * fixup,
        int64_t ncols_x,
        int64_t stride_row_x,
        int64_t nrows_x,
        int64_t ncols_dst,
        int nsm,
        bool use_stream_k,
        cudaStream_t stream) {
    int mmq_x_best = 8;
    int ntiles_x_best = INT_MAX;
    constexpr int cc = 860;
    constexpr int warp_size = 32;
    constexpr int nwarps = 8;
    constexpr int mmq_y = 128;
    const size_t smpbo = gp_cuda_info_state.devices[gp_cuda_current_device].smpbo;
    for (int mmq_x = 8; mmq_x <= 128 && ntiles_x_best > 1; mmq_x += 8) {
        const int granularity = mmq_get_granularity_host(mmq_x, cc);
        if (mmq_x % granularity != 0) continue;
        if (mmq_get_nbytes_shared<type>(mmq_x, mmq_y, cc, warp_size, nwarps) > smpbo) continue;
        const int ntiles_x = (int)((ncols_dst + mmq_x - 1) / mmq_x);
        if (ntiles_x < ntiles_x_best) {
            mmq_x_best = mmq_x;
            ntiles_x_best = ntiles_x;
        }
    }
    switch (mmq_x_best) {
        case 8:   return gp_launch_mmq_choose_stream<type, 8>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 16:  return gp_launch_mmq_choose_stream<type, 16>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 24:  return gp_launch_mmq_choose_stream<type, 24>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 32:  return gp_launch_mmq_choose_stream<type, 32>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 40:  return gp_launch_mmq_choose_stream<type, 40>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 48:  return gp_launch_mmq_choose_stream<type, 48>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 56:  return gp_launch_mmq_choose_stream<type, 56>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 64:  return gp_launch_mmq_choose_stream<type, 64>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 72:  return gp_launch_mmq_choose_stream<type, 72>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 80:  return gp_launch_mmq_choose_stream<type, 80>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 88:  return gp_launch_mmq_choose_stream<type, 88>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 96:  return gp_launch_mmq_choose_stream<type, 96>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 104: return gp_launch_mmq_choose_stream<type, 104>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 112: return gp_launch_mmq_choose_stream<type, 112>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 120: return gp_launch_mmq_choose_stream<type, 120>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        default:  return gp_launch_mmq_choose_stream<type, 128>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
    }
}

GP_CUDA_EXPORT int gp_mmq_quantize(
        int dtype,
        const float * src,
        void * q8_scratch,
        int64_t ncols_x,
        int64_t ncols_dst,
        void * stream_ptr) {
    switch ((ggml_type)dtype) {
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q5_0:
            break;
        default:
            return (int)cudaErrorInvalidValue;
    }
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    const int64_t ne0 = GGML_PAD(ncols_x, MATRIX_ROW_PADDING);
    quantize_mmq_q8_1_cuda(
        src, nullptr, q8_scratch, (ggml_type)dtype,
        ncols_x, ncols_x, 0, 0,
        ne0, ncols_dst, 1, 1,
        stream);
    return (int)cudaPeekAtLastError();
}

GP_CUDA_EXPORT int gp_mmq_matmul_q8(
        int dtype,
        const void * weights,
        float * dst,
        void * q8_scratch,
        void * fixup_scratch,
        int64_t ncols_x,
        int64_t stride_row_x,
        int64_t nrows_x,
        int64_t ncols_dst,
        void * stream_ptr) {
    cudaStream_t stream = (cudaStream_t) stream_ptr;
    const int nsm = gp_cuda_info_state.devices[gp_cuda_current_device].nsm;
    const bool use_stream_k = ncols_dst <= 128 || ncols_dst >= 512;
    cudaError_t err;
    switch ((ggml_type) dtype) {
        case GGML_TYPE_Q4_K:
            err = gp_launch_mmq_type<GGML_TYPE_Q4_K>(
                (const char *) weights, (const int *) q8_scratch, dst, (float *) fixup_scratch,
                ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
            break;
        case GGML_TYPE_Q5_K:
            err = gp_launch_mmq_type<GGML_TYPE_Q5_K>(
                (const char *) weights, (const int *) q8_scratch, dst, (float *) fixup_scratch,
                ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
            break;
        case GGML_TYPE_Q6_K:
            err = gp_launch_mmq_type<GGML_TYPE_Q6_K>(
                (const char *) weights, (const int *) q8_scratch, dst, (float *) fixup_scratch,
                ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
            break;
        case GGML_TYPE_Q8_0:
            err = gp_launch_mmq_type<GGML_TYPE_Q8_0>(
                (const char *) weights, (const int *) q8_scratch, dst, (float *) fixup_scratch,
                ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
            break;
        case GGML_TYPE_Q5_0:
            err = gp_launch_mmq_type<GGML_TYPE_Q5_0>(
                (const char *) weights, (const int *) q8_scratch, dst, (float *) fixup_scratch,
                ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
            break;
        default:
            return (int) cudaErrorInvalidValue;
    }
    return (int) err;
}

GP_CUDA_EXPORT int gp_mmq_matmul(
        int dtype,
        const void * weights,
        const float * src,
        float * dst,
        void * q8_scratch,
        void * fixup_scratch,
        int64_t ncols_x,
        int64_t stride_row_x,
        int64_t nrows_x,
        int64_t ncols_dst,
        void * stream_ptr) {
    int err = gp_mmq_quantize(
        dtype, src, q8_scratch, ncols_x, ncols_dst, stream_ptr);
    if (err != 0) return err;
    return gp_mmq_matmul_q8(
        dtype, weights, dst, q8_scratch, fixup_scratch,
        ncols_x, stride_row_x, nrows_x, ncols_dst, stream_ptr);
}

GP_CUDA_EXPORT int gp_f32_matmul(
        void * blas,
        const float * weights,
        const float * src,
        float * dst,
        int cols,
        int output_rows,
        int input_rows) {
    if (blas == nullptr || weights == nullptr || src == nullptr || dst == nullptr ||
        cols <= 0 || output_rows <= 0 || input_rows <= 0) {
        return (int) cudaErrorInvalidValue;
    }
    cublasHandle_t handle = (cublasHandle_t) blas;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        output_rows,
        input_rows,
        cols,
        &alpha,
        weights,
        cols,
        src,
        cols,
        &beta,
        dst,
        output_rows);
    return status == CUBLAS_STATUS_SUCCESS ? 0 : 10000 + (int) status;
}

__global__ void gp_f32_matvec_kernel(
        const float * __restrict__ weights,
        const float * __restrict__ src,
        float * __restrict__ dst,
        int cols) {
    __shared__ float partial[256];
    const int output_row = (int) blockIdx.x;
    const float * row = weights + (int64_t) output_row * cols;
    float sum = 0.0f;
    for (int col = (int) threadIdx.x; col < cols; col += (int) blockDim.x) {
        sum += row[col] * src[col];
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = (int) blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int) threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        dst[output_row] = partial[0];
    }
}

GP_CUDA_EXPORT int gp_f32_matvec(
        const float * weights,
        const float * src,
        float * dst,
        int cols,
        int output_rows,
        void * stream) {
    if (weights == nullptr || src == nullptr || dst == nullptr ||
        cols <= 0 || output_rows <= 0) {
        return (int) cudaErrorInvalidValue;
    }
    gp_f32_matvec_kernel<<<output_rows, 256, 0, (cudaStream_t) stream>>>(
        weights, src, dst, cols);
    return (int) cudaPeekAtLastError();
}

GP_CUDA_EXPORT int gp_mmvq_quantize(
        const float * src,
        void * q8_scratch,
        int64_t ncols_x,
        void * stream_ptr) {
    if (ncols_x <= 0 || ncols_x > (int64_t)INT_MAX) {
        return (int)cudaErrorInvalidValue;
    }
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    const int64_t ne0 = GGML_PAD(ncols_x, MATRIX_ROW_PADDING);
    quantize_row_q8_1_cuda(
        src, nullptr, q8_scratch, GGML_TYPE_Q4_K,
        ncols_x, ncols_x, 0, 0,
        ne0, 1, 1, 1,
        stream);
    return (int)cudaPeekAtLastError();
}

GP_CUDA_EXPORT int gp_mmvq_matvec_q8(
        int dtype,
        const void * weights,
        float * dst,
        void * q8_scratch,
        int64_t ncols_x,
        int64_t stride_row_x,
        int64_t nrows_x,
        void * stream_ptr) {
    switch ((ggml_type) dtype) {
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q5_0:
            break;
        default:
            return (int) cudaErrorInvalidValue;
    }
    if (ncols_x <= 0 || nrows_x <= 0 ||
        ncols_x > (int64_t) INT_MAX ||
        nrows_x > (int64_t) INT_MAX ||
        stride_row_x > (int64_t) INT_MAX) {
        return (int) cudaErrorInvalidValue;
    }

    cudaStream_t stream = (cudaStream_t) stream_ptr;
    const int64_t ne0 = GGML_PAD(ncols_x, MATRIX_ROW_PADDING);
    ggml_cuda_mm_fusion_args_device fusion{};
    mul_mat_vec_q_switch_type(
        weights, (ggml_type) dtype, q8_scratch, nullptr, fusion, dst,
        (int) ncols_x, (int) nrows_x, 1,
        (int) stride_row_x, (int) (ne0 / QK8_1), (int) nrows_x,
        1, 1, 1,
        1, 1, 1,
        1, 1, 1, 1, 1,
        0, stream);
    return (int) cudaPeekAtLastError();
}

GP_CUDA_EXPORT int gp_mmvq_matvec(
        int dtype,
        const void * weights,
        const float * src,
        float * dst,
        void * q8_scratch,
        int64_t ncols_x,
        int64_t stride_row_x,
        int64_t nrows_x,
        void * stream_ptr) {
    int err = gp_mmvq_quantize(src, q8_scratch, ncols_x, stream_ptr);
    if (err != cudaSuccess) return err;
    return gp_mmvq_matvec_q8(dtype, weights, dst, q8_scratch, ncols_x,
        stride_row_x, nrows_x, stream_ptr);
}
