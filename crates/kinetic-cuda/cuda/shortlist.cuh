#pragma once

#include "common.cuh"
#include "gemv_kquant.cuh"
#include "gemv_q8_0.cuh"

// GPU-resident exact two-stage Shortlist screening for single-vector greedy
// decoding (`Q8_0` and `Q6_K` output heads). Reuses `GemvQ8_0` / `KqQ6K` row
// evaluators bit-for-bit over surviving candidate rows inside one CUDA graph.

template <class F>
__device__ __forceinline__ float shortlist_dot_scalar(const uint8_t* w, size_t total, unsigned row,
                                                      unsigned blocks_per_row, Q8Input x) {
    const unsigned lane = threadIdx.x % WARP;
    const unsigned part = lane % F::LANES;
    const size_t first = static_cast<size_t>(row) * blocks_per_row;
    float sum = 0.0f;
    for (unsigned b = lane / F::LANES; b < blocks_per_row; b += WARP / F::LANES) {
        const typename F::Block blk = F::load(w, total, first + b, part);
        sum = __fadd_rn(sum, F::dot(blk, b, part, x));
    }
    return warp_sum(sum);
}

template <class F>
__device__ __forceinline__ float shortlist_dot_kq(const uint8_t* w, size_t total, unsigned row,
                                                  unsigned blocks_per_row, Q8Input x) {
    const unsigned lane = threadIdx.x % WARP, v = lane / 8, j = lane % 8;
    const size_t first = static_cast<size_t>(row) * blocks_per_row;
    float acc = 0.0f;
    for (unsigned base = 0; base < blocks_per_row; base += KQ_SPLIT) {
        const unsigned b = base + v;
        const bool active = b < blocks_per_row;
        typename F::Block blk = {};
        if (active) blk = F::load(w, total, first + b, j);
        float t = F::term(blk, active ? b : 0, j, x);
#pragma unroll
        for (unsigned o = 1; o < 8; o *= 2) t = __fadd_rn(t, __shfl_xor_sync(FULL_MASK, t, o));
        if (active) acc = __fadd_rn(acc, t);
    }
    float s = __shfl_sync(FULL_MASK, acc, 0);
#pragma unroll
    for (unsigned p = 1; p < KQ_SPLIT; ++p) s = __fadd_rn(s, __shfl_sync(FULL_MASK, acc, 8 * p));
    return s;
}

__device__ __forceinline__ float q8_input_norm(Q8Input x, unsigned q8_blocks) {
    const unsigned lane = threadIdx.x % WARP;
    float sum_sq = 0.0f;
    for (unsigned b = lane; b < q8_blocks; b += WARP) {
        const float d = __ldg(x.ds + b).x;
        const Words a = load_cached(x.qs + b * 32);
        const Words c = load_cached(x.qs + b * 32 + 16);
        const int qq = dot16(a, a, dot16(c, c, 0));
        sum_sq += d * d * static_cast<float>(qq);
    }
    sum_sq = warp_sum(sum_sq);
    // Slight margin covers f32 accumulation variance across blocks.
    return sqrtf(sum_sq) * 1.00001f;
}

template <class DotFn>
__device__ __forceinline__ void shortlist_prepare_impl(const uint8_t* head, size_t total,
                                                       unsigned blocks_per_row, unsigned q8_blocks,
                                                       Q8Input x, const unsigned* approx_best,
                                                       float* d_floor, float* d_norm,
                                                       unsigned* d_count, DotFn dot_row) {
    const unsigned r0 = *approx_best;
    const float floor_val = dot_row(head, total, r0, blocks_per_row, x);
    const float norm_val = q8_input_norm(x, q8_blocks);
    if (threadIdx.x == 0) {
        *d_floor = floor_val;
        *d_norm = norm_val;
        *d_count = 0;
    }
}

extern "C" __global__ void __launch_bounds__(WARP)
    shortlist_prepare_q8_0(const uint8_t* head, size_t total, unsigned blocks_per_row,
                           unsigned q8_blocks, const int8_t* xq, const float2* xds,
                           const unsigned* approx_best, float* d_floor, float* d_norm,
                           unsigned* d_count) {
    shortlist_prepare_impl(head, total, blocks_per_row, q8_blocks, Q8Input{xq, xds}, approx_best,
                           d_floor, d_norm, d_count, shortlist_dot_scalar<GemvQ8_0>);
}

extern "C" __global__ void __launch_bounds__(WARP)
    shortlist_prepare_q6_k(const uint8_t* head, size_t total, unsigned blocks_per_row,
                           unsigned q8_blocks, const int8_t* xq, const float2* xds,
                           const unsigned* approx_best, float* d_floor, float* d_norm,
                           unsigned* d_count) {
    shortlist_prepare_impl(head, total, blocks_per_row, q8_blocks, Q8Input{xq, xds}, approx_best,
                           d_floor, d_norm, d_count, shortlist_dot_kq<KqQ6K>);
}

extern "C" __global__ void __launch_bounds__(256)
    shortlist_filter(const float* approx, const float2* radii, unsigned vocab, float slack,
                     const unsigned* approx_best, const float* d_floor, const float* d_norm,
                     unsigned* d_count, unsigned* kept_idx) {
    const float floor_val = *d_floor;
    const float norm_val = *d_norm;
    const float slack_norm = slack * norm_val;
    const unsigned r0 = *approx_best;
    const unsigned r = blockIdx.x * blockDim.x + threadIdx.x;

    bool keep = false;
    if (r < vocab) {
        if (r == r0 || !isfinite(floor_val)) {
            keep = true;
        } else {
            const float a = approx[r];
            const float2 rad = __ldg(radii + r);
            const float bound = a + rad.x * norm_val + rad.y * slack_norm;
            keep = !(bound < floor_val);
        }
    }

    const unsigned mask = __ballot_sync(FULL_MASK, keep);
    if (mask) {
        const unsigned lane = threadIdx.x % WARP;
        const unsigned leader = __ffs(mask) - 1;
        unsigned base = 0;
        if (lane == leader) base = atomicAdd(d_count, __popc(mask));
        base = __shfl_sync(FULL_MASK, base, leader);
        if (keep) {
            const unsigned rank = __popc(mask & ((1u << lane) - 1));
            kept_idx[base + rank] = r;
        }
    }
}

template <class DotFn>
__device__ __forceinline__ void shortlist_eval_impl(const uint8_t* head, size_t total,
                                                    unsigned blocks_per_row, Q8Input x,
                                                    const unsigned* d_count,
                                                    const unsigned* kept_idx, float* kept_val,
                                                    DotFn dot_row) {
    const unsigned m = *d_count;
    const unsigned wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    const unsigned total_warps = gridDim.x * (blockDim.x / WARP);
    const unsigned lane = threadIdx.x % WARP;
    for (unsigned idx = wid; idx < m; idx += total_warps) {
        const unsigned row = kept_idx[idx];
        const float s = dot_row(head, total, row, blocks_per_row, x);
        if (lane == 0) kept_val[idx] = s;
    }
}

extern "C" __global__ void __launch_bounds__(128)
    shortlist_eval_q8_0(const uint8_t* head, size_t total, unsigned blocks_per_row,
                        const int8_t* xq, const float2* xds, const unsigned* d_count,
                        const unsigned* kept_idx, float* kept_val) {
    shortlist_eval_impl(head, total, blocks_per_row, Q8Input{xq, xds}, d_count, kept_idx, kept_val,
                        shortlist_dot_scalar<GemvQ8_0>);
}

extern "C" __global__ void __launch_bounds__(128)
    shortlist_eval_q6_k(const uint8_t* head, size_t total, unsigned blocks_per_row,
                        const int8_t* xq, const float2* xds, const unsigned* d_count,
                        const unsigned* kept_idx, float* kept_val) {
    shortlist_eval_impl(head, total, blocks_per_row, Q8Input{xq, xds}, d_count, kept_idx, kept_val,
                        shortlist_dot_kq<KqQ6K>);
}

extern "C" __global__ void __launch_bounds__(256)
    shortlist_argmax(const unsigned* d_count, const unsigned* kept_idx, const float* kept_val,
                     unsigned* out_best) {
    __shared__ float smem_v[8];
    __shared__ unsigned smem_r[8];

    const unsigned m = *d_count;
    float best_v = -CUDART_INF_F;
    unsigned best_r = 0xffffffffu;

    for (unsigned idx = threadIdx.x; idx < m; idx += 256) {
        const float v = kept_val[idx];
        const unsigned r = kept_idx[idx];
        if (best_r == 0xffffffffu || v > best_v || (v == best_v && r < best_r)) {
            best_v = v;
            best_r = r;
        }
    }

    const unsigned lane = threadIdx.x % WARP;
    const unsigned warp = threadIdx.x / WARP;
#pragma unroll
    for (unsigned o = WARP / 2; o > 0; o /= 2) {
        const float ov = __shfl_xor_sync(FULL_MASK, best_v, o);
        const unsigned orow = __shfl_xor_sync(FULL_MASK, best_r, o);
        if (orow != 0xffffffffu &&
            (best_r == 0xffffffffu || ov > best_v || (ov == best_v && orow < best_r))) {
            best_v = ov;
            best_r = orow;
        }
    }
    if (lane == 0) {
        smem_v[warp] = best_v;
        smem_r[warp] = best_r;
    }
    __syncthreads();

    if (warp == 0) {
        best_v = (lane < 8) ? smem_v[lane] : -CUDART_INF_F;
        best_r = (lane < 8) ? smem_r[lane] : 0xffffffffu;
#pragma unroll
        for (unsigned o = 4; o > 0; o /= 2) {
            const float ov = __shfl_xor_sync(FULL_MASK, best_v, o);
            const unsigned orow = __shfl_xor_sync(FULL_MASK, best_r, o);
            if (orow != 0xffffffffu &&
                (best_r == 0xffffffffu || ov > best_v || (ov == best_v && orow < best_r))) {
                best_v = ov;
                best_r = orow;
            }
        }
        if (lane == 0) *out_best = best_r;
    }
}

