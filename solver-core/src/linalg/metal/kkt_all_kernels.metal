// kkt_all_kernels.metal
//
// One-file Metal kernel pack for HSDE/SCS-style convex optimization solvers that want
// a cuDSS-like sparse direct KKT solve workflow on Apple GPUs.
//
// This file merges and extends:
//   - hsde_metal_kernels.metal                  (SpMV, vector ops, SOC proj, reductions)
//   - sparse_direct_metal_kernels.metal         (permutes, D scaling, level-scheduled SpTRSV)
//   - metal_cudss_like_ldlt_kernels_v3.metal    (dense/supernodal LDLT/TRSM/update incl. SIMD-group MMA)
//
// The intent is that your Rust solver orchestrates the phases:
//
//   ANALYSIS (CPU):
//     - ordering / symbolic factorization / supernodes (or CSR solve levels)
//     - build schedules and mapping tables
//     - allocate GPU buffers
//
//   FACTORIZATION (GPU-heavy):
//     - dense LDLᵀ on diagonal blocks (no pivoting; quasi-definite KKT recommended)
//     - TRSM on off-diagonal panels
//     - Schur complement updates via SYRK/GEMM-style kernels (portable or SIMD-group MMA)
//
//   SOLVE (GPU-heavy):
//     - permute RHS
//     - forward solve (SpTRSV by levels, or supernodal block solve)
//     - diagonal scaling by D^{-1}
//     - backward solve
//     - inverse permute
//
// Additionally, for iterative refinement and HSDE outer loops, this file includes:
//   - CSR SpMV kernels
//   - vector ops
//   - dot/reduction building blocks
//
// Optional GPU assembly helpers are included at the end (scatter-add / gather). These are
// intentionally generic and assume **no index collisions** per call unless you use the atomic variant.
//
// NOTE ON FLOAT ATOMICS:
//   This file provides a non-atomic scatter-add kernel that is correct only if the index map has
//   no duplicates. If you need true atomic float accumulation, you will need to either:
//   - ensure your target supports atomic_float (Metal 3+), or
//   - restructure assembly as a gather (no collisions), or
//   - split updates so each destination entry is written by one thread.
//
// --------------------------------------------------------------------------------------
// Conventions
//
// - Float type: float32
// - Indices: uint32
//
// - CSR uses:
//     values[nnz], col_indices[nnz], row_ptr[m+1]
//
// - Triangular solves (SpTRSV) use CSR with implicit unit diagonal:
//     L: strictly-lower in CSR (col < row), diag not stored
//     U: strictly-upper in CSR (col > row), diag not stored
//
// - Dense matrices are row-major with leading dimension (ld) in elements.
//
// --------------------------------------------------------------------------------------

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// -----------------------------------------
// Config macros
// -----------------------------------------

#ifndef HSDE_TG_SIZE
#define HSDE_TG_SIZE 256
#endif

// -----------------------------------------
// 1) Sparse kernels: CSR SpMV + reductions
// -----------------------------------------

kernel void csr_spmv_f32(
    device const float* values        [[buffer(0)]],
    device const uint*  col_indices   [[buffer(1)]],
    device const uint*  row_ptr       [[buffer(2)]],
    device const float* x             [[buffer(3)]],
    device float*       y             [[buffer(4)]],
    constant uint&      m             [[buffer(5)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= m) return;

    const uint start = row_ptr[row];
    const uint end   = row_ptr[row + 1];

    float acc = 0.0f;
    for (uint idx = start; idx < end; idx++) {
        const uint col = col_indices[idx];
        acc += values[idx] * x[col];
    }
    y[row] = acc;
}

// y = add + A*x
kernel void csr_spmv_add_f32(
    device const float* values        [[buffer(0)]],
    device const uint*  col_indices   [[buffer(1)]],
    device const uint*  row_ptr       [[buffer(2)]],
    device const float* x             [[buffer(3)]],
    device const float* add           [[buffer(4)]],
    device float*       y             [[buffer(5)]],
    constant uint&      m             [[buffer(6)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= m) return;

    const uint start = row_ptr[row];
    const uint end   = row_ptr[row + 1];

    float acc = add[row];
    for (uint idx = start; idx < end; idx++) {
        const uint col = col_indices[idx];
        acc += values[idx] * x[col];
    }
    y[row] = acc;
}

// out[row] = sum(values[idx]^2 for idx in row)
kernel void csr_row_sumsquares_f32(
    device const float* values   [[buffer(0)]],
    device const uint*  row_ptr  [[buffer(1)]],
    device float*       out      [[buffer(2)]],
    constant uint&      m        [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= m) return;

    const uint start = row_ptr[row];
    const uint end   = row_ptr[row + 1];

    float acc = 0.0f;
    for (uint idx = start; idx < end; idx++) {
        const float v = values[idx];
        acc += v * v;
    }
    out[row] = acc;
}

// dot partial: partial[group] = sum a[i]*b[i] over a strided subset
kernel void dot_partial_f32(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device float*       partial [[buffer(2)]],
    constant uint&      n       [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tpg [[threads_per_grid]],
    uint3 tgpos [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch[HSDE_TG_SIZE];

    const uint stride = tpg.x;
    const uint gid_x = gid.x;

    float sum = 0.0f;
    for (uint i = gid_x; i < n; i += stride) {
        sum += a[i] * b[i];
    }

    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = HSDE_TG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            scratch[tid] += scratch[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        partial[tgpos.x] = scratch[0];
    }
}

// reduce sum partial: out_partial[group] = sum in_partial[...] (multi-pass)
kernel void reduce_sum_partial_f32(
    device const float* in_partial  [[buffer(0)]],
    device float*       out_partial [[buffer(1)]],
    constant uint&      n           [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tpg [[threads_per_grid]],
    uint3 tgpos [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch[HSDE_TG_SIZE];

    const uint stride = tpg.x;
    const uint gid_x = gid.x;

    float sum = 0.0f;
    for (uint i = gid_x; i < n; i += stride) {
        sum += in_partial[i];
    }

    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = HSDE_TG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            scratch[tid] += scratch[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        out_partial[tgpos.x] = scratch[0];
    }
}

// -----------------------------------------
// 2) Vector kernels (HSDE / refinement)
// -----------------------------------------

kernel void vec_add_f32(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant uint&      n   [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = a[gid] + b[gid];
}

kernel void vec_sub_f32(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant uint&      n   [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = a[gid] - b[gid];
}

kernel void vec_copy_f32(
    device const float* x   [[buffer(0)]],
    device float*       y   [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    y[gid] = x[gid];
}

kernel void vec_set_f32(
    device float*  x      [[buffer(0)]],
    constant float& value [[buffer(1)]],
    constant uint&  n     [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    x[gid] = value;
}

kernel void vec_scale_inplace_f32(
    device float*   x      [[buffer(0)]],
    constant float& alpha  [[buffer(1)]],
    constant uint&  n      [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    x[gid] *= alpha;
}

kernel void vec_axpy_inplace_f32(
    device const float* x     [[buffer(0)]],
    device float*       y     [[buffer(1)]],
    constant float&     alpha [[buffer(2)]],
    constant uint&      n     [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    y[gid] += alpha * x[gid];
}

// y = x + alpha*y  (xpay)
kernel void vec_xpay_inplace_f32(
    device const float* x     [[buffer(0)]],
    device float*       y     [[buffer(1)]],
    constant float&     alpha [[buffer(2)]],
    constant uint&      n     [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    y[gid] = x[gid] + alpha * y[gid];
}

// y = max(y, 0)
kernel void proj_nonneg_inplace_f32(
    device float*  y   [[buffer(0)]],
    constant uint& n   [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    y[gid] = max(y[gid], 0.0f);
}

// -----------------------------------------
// 3) Cone projection: SOC blocks (optional)
// -----------------------------------------

struct SocBlock {
    uint offset;
    uint length;
};

kernel void proj_soc_inplace_f32(
    device float*           y          [[buffer(0)]],
    device const SocBlock*  blocks     [[buffer(1)]],
    constant uint&          num_blocks [[buffer(2)]],
    uint bid [[thread_position_in_grid]]
) {
    if (bid >= num_blocks) return;

    const uint off = blocks[bid].offset;
    const uint len = blocks[bid].length;
    if (len < 2) return;

    const float t = y[off + 0];

    float ss = 0.0f;
    for (uint i = 1; i < len; i++) {
        const float v = y[off + i];
        ss += v * v;
    }
    const float norm_x = sqrt(ss);

    if (norm_x <= t) return;

    if (norm_x <= -t) {
        for (uint i = 0; i < len; i++) {
            y[off + i] = 0.0f;
        }
        return;
    }

    const float out_t = 0.5f * (t + norm_x);
    const float denom = max(norm_x, 1.0e-20f);
    const float scale = out_t / denom;

    y[off + 0] = out_t;
    for (uint i = 1; i < len; i++) {
        y[off + i] *= scale;
    }
}

// -----------------------------------------
// 4) Solve-phase kernels: permutations + SpTRSV
// -----------------------------------------

// out[i] = in[perm[i]]
kernel void permute_gather_f32(
    device const float* in      [[buffer(0)]],
    device const uint*  perm    [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      n       [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = in[perm[gid]];
}

// Alias name used in some docs: y = P^T x (if perm is that mapping)
kernel void permute_vec_f32(
    device const float* in      [[buffer(0)]],
    device const uint*  perm    [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      n       [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = in[perm[gid]];
}

// out[perm[i]] = in[i]
kernel void permute_scatter_f32(
    device const float* in      [[buffer(0)]],
    device const uint*  perm    [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      n       [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[perm[gid]] = in[gid];
}

// Alias name: inverse permutation (depends on your perm convention)
kernel void permute_vec_inv_f32(
    device const float* in      [[buffer(0)]],
    device const uint*  perm    [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      n       [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[perm[gid]] = in[gid];
}

// x[i] *= dinv[i]
kernel void apply_dinv_inplace_f32(
    device float*       x     [[buffer(0)]],
    device const float* dinv  [[buffer(1)]],
    constant uint&      n     [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    x[gid] *= dinv[gid];
}

// One-level SpTRSV: y = L^{-1} b for rows in one level (CSR strictly-lower, unit diag)
kernel void sptrsv_lower_level_unitdiag_csr_f32(
    device const float* values       [[buffer(0)]],
    device const uint*  col_indices  [[buffer(1)]],
    device const uint*  row_ptr      [[buffer(2)]],
    device const float* b            [[buffer(3)]],
    device float*       y            [[buffer(4)]],
    device const uint*  level_rows   [[buffer(5)]],
    constant uint&      level_size   [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= level_size) return;

    const uint row = level_rows[tid];

    const uint start = row_ptr[row];
    const uint end   = row_ptr[row + 1];

    float acc = b[row];
    for (uint idx = start; idx < end; idx++) {
        const uint col = col_indices[idx];
        acc -= values[idx] * y[col];
    }
    y[row] = acc;
}

// Alias name matching the screenshot shorthand
kernel void sptrsv_lower_level_f32(
    device const float* values       [[buffer(0)]],
    device const uint*  col_indices  [[buffer(1)]],
    device const uint*  row_ptr      [[buffer(2)]],
    device const float* b            [[buffer(3)]],
    device float*       y            [[buffer(4)]],
    device const uint*  level_rows   [[buffer(5)]],
    constant uint&      level_size   [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= level_size) return;

    const uint row = level_rows[tid];

    const uint start = row_ptr[row];
    const uint end   = row_ptr[row + 1];

    float acc = b[row];
    for (uint idx = start; idx < end; idx++) {
        const uint col = col_indices[idx];
        acc -= values[idx] * y[col];
    }
    y[row] = acc;
}

// One-level SpTRSV: x = U^{-1} b for rows in one level (CSR strictly-upper, unit diag)
kernel void sptrsv_upper_level_unitdiag_csr_f32(
    device const float* values       [[buffer(0)]],
    device const uint*  col_indices  [[buffer(1)]],
    device const uint*  row_ptr      [[buffer(2)]],
    device const float* b            [[buffer(3)]],
    device float*       x            [[buffer(4)]],
    device const uint*  level_rows   [[buffer(5)]],
    constant uint&      level_size   [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= level_size) return;

    const uint row = level_rows[tid];

    const uint start = row_ptr[row];
    const uint end   = row_ptr[row + 1];

    float acc = b[row];
    for (uint idx = start; idx < end; idx++) {
        const uint col = col_indices[idx];
        acc -= values[idx] * x[col];
    }
    x[row] = acc;
}

// Alias name matching the screenshot shorthand
kernel void sptrsv_upper_level_f32(
    device const float* values       [[buffer(0)]],
    device const uint*  col_indices  [[buffer(1)]],
    device const uint*  row_ptr      [[buffer(2)]],
    device const float* b            [[buffer(3)]],
    device float*       x            [[buffer(4)]],
    device const uint*  level_rows   [[buffer(5)]],
    constant uint&      level_size   [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= level_size) return;

    const uint row = level_rows[tid];

    const uint start = row_ptr[row];
    const uint end   = row_ptr[row + 1];

    float acc = b[row];
    for (uint idx = start; idx < end; idx++) {
        const uint col = col_indices[idx];
        acc -= values[idx] * x[col];
    }
    x[row] = acc;
}

// -----------------------------------------
// 5) Dense/supernodal numeric kernels (LDLT/TRSM/update)
// -----------------------------------------

// Force inlining helper for hot functions.
#define METAL_FUNC inline __attribute__((always_inline))

using sgf8x8 = simdgroup_matrix<float, 8, 8>;

// Descriptors
struct DenseLDLTDesc {
    uint A_offset;   // base offset (in floats) of the dense matrix in 'A' (row-major)
    uint D_offset;   // base offset (in floats) of the diagonal array in 'D' (length n)
    uint n;          // matrix dimension
    uint ld;         // leading dimension (row-major), >= n
    float pivot_min; // minimum |pivot| clamp (0 disables)
};

struct DenseTRSMRightUnitUpperDesc {
    uint L_offset;
    uint B_offset;
    uint m;    // rows of B
    uint n;    // cols of B (and dimension of L)
    uint ldL;  // >= n
    uint ldB;  // >= n
};

struct DenseColScaleDesc {
    uint B_offset;
    uint diag_offset;
    uint m;
    uint n;
    uint ldB;
    uint mode; // 0 multiply, 1 multiply by reciprocal
};

struct DenseSYRKLDLTUpdateDesc {
    uint C_offset;
    uint B_offset;
    uint D_offset;
    uint m;   // C is m x m, B is m x n
    uint n;
    uint ldC; // >= m
    uint ldB; // >= n
};

constexpr uint LDLT_N_MAX = 64;

static inline float ldlt_clamp_pivot(float d, float pivot_min) {
    if (pivot_min <= 0.0f) return d;
    const float ad = metal::abs(d);
    if (ad >= pivot_min) return d;
    return metal::copysign(pivot_min, (d == 0.0f ? 1.0f : d));
}

// Dense LDLᵀ (no pivoting), batched by desc array.
//
// Output convention:
// - A overwritten to store L in strict-lower triangle; diagonal set to 1
// - D written as diagonal entries (length n)
//
// Fast path uses threadgroup memory for n<=64.
kernel void dense_ldlt_nopivot_inplace_f32_batched(
    device float* A                  [[buffer(0)]],
    device float* D                  [[buffer(1)]],
    constant DenseLDLTDesc* descs    [[buffer(2)]],
    uint tid                         [[thread_index_in_threadgroup]],
    uint tgsz                        [[threads_per_threadgroup]],
    uint3 tgp                        [[threadgroup_position_in_grid]])
{
    const uint mat_id = tgp.x;
    const DenseLDLTDesc ds = descs[mat_id];

    const uint n = ds.n;
    const uint ld = ds.ld;
    const uint baseA = ds.A_offset;
    const uint baseD = ds.D_offset;
    const float pivot_min = ds.pivot_min;

    if (n == 0u) return;

    if (n <= LDLT_N_MAX) {
        threadgroup float sA[LDLT_N_MAX * LDLT_N_MAX];
        threadgroup float sD[LDLT_N_MAX];

        // Load A into shared memory (top-left n x n)
        for (uint idx = tid; idx < n * n; idx += tgsz) {
            const uint r = idx / n;
            const uint c = idx - r * n;
            sA[r * LDLT_N_MAX + c] = A[baseA + r * ld + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Unblocked LDLᵀ (no pivot)
        for (uint k = 0; k < n; ++k) {
            if (tid == 0u) {
                float dkk = ldlt_clamp_pivot(sA[k * LDLT_N_MAX + k], pivot_min);
                sD[k] = dkk;
                sA[k * LDLT_N_MAX + k] = 1.0f; // unit diag for L
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const float dkk = sD[k];
            const float invd = 1.0f / dkk;

            // L(i,k) = A(i,k)/dkk for i>k
            for (uint i = k + 1u + tid; i < n; i += tgsz) {
                sA[i * LDLT_N_MAX + k] *= invd;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // trailing update: A(i,j) -= dkk * L(i,k) * L(j,k)
            const uint rem = n - (k + 1u);
            const uint work = rem * rem;
            for (uint idx = tid; idx < work; idx += tgsz) {
                const uint ii = idx / rem;
                const uint jj = idx - ii * rem;
                const uint i = (k + 1u) + ii;
                const uint j = (k + 1u) + jj;
                const float lik = sA[i * LDLT_N_MAX + k];
                const float ljk = sA[j * LDLT_N_MAX + k];
                const float aij = sA[i * LDLT_N_MAX + j];
                sA[i * LDLT_N_MAX + j] = fma(-(lik * dkk), ljk, aij);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Store D
        for (uint i = tid; i < n; i += tgsz) {
            D[baseD + i] = sD[i];
        }

        // Store A back
        for (uint idx = tid; idx < n * n; idx += tgsz) {
            const uint r = idx / n;
            const uint c = idx - r * n;
            A[baseA + r * ld + c] = sA[r * LDLT_N_MAX + c];
        }

        return;
    }

    // Slow fallback for n>64 (single thread). Prefer blocking/panelization in orchestrator.
    if (tid != 0u) return;

    for (uint k = 0; k < n; ++k) {
        float dkk = ldlt_clamp_pivot(A[baseA + k * ld + k], pivot_min);
        D[baseD + k] = dkk;
        A[baseA + k * ld + k] = 1.0f;

        for (uint i = k + 1u; i < n; ++i) {
            const uint ik = baseA + i * ld + k;
            A[ik] = A[ik] / dkk;
        }

        for (uint i = k + 1u; i < n; ++i) {
            const float lik = A[baseA + i * ld + k];
            const float lik_d = lik * dkk;
            for (uint j = k + 1u; j < n; ++j) {
                const float ljk = A[baseA + j * ld + k];
                const uint ij = baseA + i * ld + j;
                A[ij] = fma(-lik_d, ljk, A[ij]);
            }
        }
    }
}

// TRSM (right solve): B := B * inv(Lᵀ)
// L unit-lower (n x n), row-major. B row-major (m x n).
kernel void dense_trsm_right_unit_upper_from_unit_lower_f32(
    device const float* L                    [[buffer(0)]],
    device float*       B                    [[buffer(1)]],
    constant DenseTRSMRightUnitUpperDesc& ds [[buffer(2)]],
    uint gid                                 [[thread_position_in_grid_x]]
) {
    if (gid >= ds.m) return;

    const uint n   = ds.n;
    const uint ldL = ds.ldL;
    const uint ldB = ds.ldB;
    const uint row = gid;

    // Solve x * Lᵀ = b  =>  x_k = b_k - sum_{j<k} x_j * L(k,j)
    for (uint k = 0; k < n; ++k) {
        float x = B[ds.B_offset + row * ldB + k];
        for (uint j = 0; j < k; ++j) {
            const float lkj = L[ds.L_offset + k * ldL + j];
            x = fma(-B[ds.B_offset + row * ldB + j], lkj, x);
        }
        B[ds.B_offset + row * ldB + k] = x;
    }
}

// Cached TRSM: caches L into threadgroup when n<=64.
kernel void dense_trsm_right_unit_upper_from_unit_lower_f32_cached(
    device const float* L                    [[buffer(0)]],
    device float*       B                    [[buffer(1)]],
    constant DenseTRSMRightUnitUpperDesc& ds [[buffer(2)]],
    uint tid                                 [[thread_index_in_threadgroup]],
    uint tgsz                                [[threads_per_threadgroup]],
    uint gid                                 [[thread_position_in_grid_x]]
) {
    const uint n = ds.n;
    if (n == 0u) return;

    if (n > 64u) {
        // fallback to naïve
        if (gid < ds.m) {
            const uint ldL = ds.ldL;
            const uint ldB = ds.ldB;
            const uint row = gid;
            for (uint k = 0; k < n; ++k) {
                float x = B[ds.B_offset + row * ldB + k];
                for (uint j = 0; j < k; ++j) {
                    const float lkj = L[ds.L_offset + k * ldL + j];
                    x = fma(-B[ds.B_offset + row * ldB + j], lkj, x);
                }
                B[ds.B_offset + row * ldB + k] = x;
            }
        }
        return;
    }

    threadgroup float sL[64 * 64];

    for (uint idx = tid; idx < n * n; idx += tgsz) {
        const uint r = idx / n;
        const uint c = idx - r * n;
        sL[r * 64 + c] = L[ds.L_offset + r * ds.ldL + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid >= ds.m) return;

    const uint ldB = ds.ldB;
    const uint row = gid;

    for (uint k = 0; k < n; ++k) {
        float x = B[ds.B_offset + row * ldB + k];

        uint j = 0;
        for (; j + 4u <= k; j += 4u) {
            const uint bidx = ds.B_offset + row * ldB + j;
            const float4 bx = float4(B[bidx + 0u], B[bidx + 1u], B[bidx + 2u], B[bidx + 3u]);
            const float4 lx = float4(sL[k * 64 + (j + 0u)],
                                     sL[k * 64 + (j + 1u)],
                                     sL[k * 64 + (j + 2u)],
                                     sL[k * 64 + (j + 3u)]);
            x = fma(-bx.x, lx.x, x);
            x = fma(-bx.y, lx.y, x);
            x = fma(-bx.z, lx.z, x);
            x = fma(-bx.w, lx.w, x);
        }
        for (; j < k; ++j) {
            x = fma(-B[ds.B_offset + row * ldB + j], sL[k * 64 + j], x);
        }

        B[ds.B_offset + row * ldB + k] = x;
    }
}

// Column scaling: B *= diag  or  B *= 1/diag
kernel void dense_col_scale_f32(
    device float*               B        [[buffer(0)]],
    device const float*         diag     [[buffer(1)]],
    constant DenseColScaleDesc& ds       [[buffer(2)]],
    uint2 gid                            [[thread_position_in_grid]]
) {
    const uint row = gid.y;
    const uint col = gid.x;
    if (row >= ds.m || col >= ds.n) return;

    float d = diag[ds.diag_offset + col];
    if (ds.mode == 1u) d = 1.0f / d;

    const uint idx = ds.B_offset + row * ds.ldB + col;
    B[idx] *= d;
}

// Portable tiled update: C := C - (B*diag(D))*Bᵀ
constexpr uint TILE_M = 16;
constexpr uint TILE_K = 16;

kernel void dense_syrk_ldlt_update_f32(
    device float* C                        [[buffer(0)]],
    device const float* B                  [[buffer(1)]],
    device const float* D                  [[buffer(2)]],
    constant DenseSYRKLDLTUpdateDesc& ds   [[buffer(3)]],
    uint3 tid                              [[thread_position_in_threadgroup]],
    uint3 tgp                              [[threadgroup_position_in_grid]]
) {
    const uint local_j = tid.x;
    const uint local_i = tid.y;

    const uint tile_j = tgp.x * TILE_M;
    const uint tile_i = tgp.y * TILE_M;

    const uint j = tile_j + local_j;
    const uint i = tile_i + local_i;

    threadgroup float sA[TILE_M * TILE_K]; // B(i,k)*D(k)
    threadgroup float sB[TILE_M * TILE_K]; // B(j,k)

    float acc = 0.0f;

    for (uint k0 = 0; k0 < ds.n; k0 += TILE_K) {
        const uint kA = k0 + local_j;
        if (i < ds.m && kA < ds.n) {
            const float bij = B[ds.B_offset + i * ds.ldB + kA];
            const float dk  = D[ds.D_offset + kA];
            sA[local_i * TILE_K + local_j] = bij * dk;
        } else {
            sA[local_i * TILE_K + local_j] = 0.0f;
        }

        const uint kB = k0 + local_i;
        if (j < ds.m && kB < ds.n) {
            sB[local_j * TILE_K + local_i] = B[ds.B_offset + j * ds.ldB + kB];
        } else {
            sB[local_j * TILE_K + local_i] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0; kk < TILE_K; ++kk) {
            acc = fma(sA[local_i * TILE_K + kk], sB[local_j * TILE_K + kk], acc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (i < ds.m && j < ds.m) {
        const uint idx = ds.C_offset + i * ds.ldC + j;
        C[idx] -= acc;
    }
}

// SIMD-group-matrix accelerated update (64x64 tiles)
// Dispatch threads_per_threadgroup MUST be (128,1,1).
constexpr int SG_BM = 64;
constexpr int SG_BN = 64;
constexpr int SG_BK = 32;
constexpr int SG_WM = 2;
constexpr int SG_WN = 2;
constexpr int SG_TGP = SG_WM * SG_WN * 32;
constexpr int SG_PAD_A = 4;
constexpr int SG_PAD_B = 4;
constexpr int SG_LDA = SG_BK + SG_PAD_A;
constexpr int SG_LDB = SG_BK + SG_PAD_B;
constexpr int SG_TM = SG_BM / (SG_WM * 8);
constexpr int SG_TN = SG_BN / (SG_WN * 8);
constexpr int SG_TM_STRIDE = 8 * SG_WM;
constexpr int SG_TN_STRIDE = 8 * SG_WN;

struct SGEMM64x64x32_Op {
    const int tm;
    const int tn;
    short sm;
    short sn;

    sgf8x8 As[SG_TM];
    sgf8x8 Bs[SG_TN];
    sgf8x8 acc[SG_TM * SG_TN];

    SGEMM64x64x32_Op(uint simd_group_id, uint simd_lane_id)
    : tm(int(8 * (simd_group_id / SG_WN))),
      tn(int(8 * (simd_group_id % SG_WN))) {

        short qid = short(simd_lane_id / 4);
        sm = short((qid & 4) + (simd_lane_id / 2) % 4);
        sn = short((qid & 2) * 2 + (simd_lane_id % 2) * 2);

        #pragma clang loop unroll(full)
        for (int i = 0; i < SG_TM * SG_TN; ++i) {
            acc[i] = sgf8x8(0.0f);
        }
    }

    METAL_FUNC void mma(const threadgroup float* As_tg, const threadgroup float* Bs_tg) {
        #pragma clang loop unroll(full)
        for (short kk = 0; kk < SG_BK; kk += 8) {

            const threadgroup float* Ap = As_tg + (tm + sm) * SG_LDA + (kk + sn);
            #pragma clang loop unroll(full)
            for (int i = 0; i < SG_TM; ++i) {
                As[i].thread_elements()[0] = Ap[0];
                As[i].thread_elements()[1] = Ap[1];
                Ap += SG_TM_STRIDE * SG_LDA;
            }

            const threadgroup float* Bp = Bs_tg + (tn + sn) * SG_LDB + (kk + sm);
            #pragma clang loop unroll(full)
            for (int j = 0; j < SG_TN; ++j) {
                Bs[j].thread_elements()[0] = Bp[0];
                Bs[j].thread_elements()[1] = Bp[SG_LDB];
                Bp += SG_TN_STRIDE * SG_LDB;
            }

            simdgroup_barrier(mem_flags::mem_none);

            #pragma clang loop unroll(full)
            for (int i = 0; i < SG_TM; ++i) {
                #pragma clang loop unroll(full)
                for (int j = 0; j < SG_TN; ++j) {
                    const int idx = i * SG_TN + j;
                    simdgroup_multiply_accumulate(acc[idx], As[i], Bs[j], acc[idx]);
                }
            }

            simdgroup_barrier(mem_flags::mem_none);
        }
    }

    METAL_FUNC void store_sub(device float* C0, uint ldC, uint base_i, uint base_j, uint m) const {
        #pragma clang loop unroll(full)
        for (int i = 0; i < SG_TM; ++i) {
            const uint row = base_i + uint(tm + i * SG_TM_STRIDE + sm);
            if (row >= m) continue;

            #pragma clang loop unroll(full)
            for (int j = 0; j < SG_TN; ++j) {
                const uint col0 = base_j + uint(tn + j * SG_TN_STRIDE + sn);
                const sgf8x8 r = acc[i * SG_TN + j];

                if (col0 < m) {
                    const uint idx0 = row * ldC + col0;
                    C0[idx0] = C0[idx0] - r.thread_elements()[0];
                }
                if (col0 + 1u < m) {
                    const uint idx1 = row * ldC + (col0 + 1u);
                    C0[idx1] = C0[idx1] - r.thread_elements()[1];
                }
            }
        }
    }
};

kernel void dense_syrk_ldlt_update_f32_simdgroup64(
    device float* C                        [[buffer(0)]],
    device const float* B                  [[buffer(1)]],
    device const float* D                  [[buffer(2)]],
    constant DenseSYRKLDLTUpdateDesc& ds   [[buffer(3)]],
    uint simd_lane_id                      [[thread_index_in_simdgroup]],
    uint simd_group_id                     [[simdgroup_index_in_threadgroup]],
    uint3 tgp                              [[threadgroup_position_in_grid]])
{
    const uint m = ds.m;
    const uint n = ds.n;
    const uint base_i = tgp.y * uint(SG_BM);
    const uint base_j = tgp.x * uint(SG_BN);

    device float* C0 = C + ds.C_offset;
    const device float* B0 = B + ds.B_offset;
    const device float* D0 = D + ds.D_offset;

    const uint thread_idx = simd_group_id * 32u + simd_lane_id;

    threadgroup float As_tg[SG_BM * SG_LDA];
    threadgroup float Bs_tg[SG_BN * SG_LDB];
    threadgroup float D_tg[SG_BK];

    SGEMM64x64x32_Op op(simd_group_id, simd_lane_id);

    for (uint k0 = 0; k0 < n; k0 += uint(SG_BK)) {

        if (thread_idx < uint(SG_BK)) {
            const uint kk = k0 + thread_idx;
            D_tg[thread_idx] = (kk < n) ? D0[kk] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        constexpr uint vec_size = 8;
        constexpr uint n_vecs = uint(SG_BK) / vec_size;     // 4
        constexpr uint bstride = uint(SG_TGP) / n_vecs;     // 32

        const uint bi = thread_idx / n_vecs;               // 0..31
        const uint bj = vec_size * (thread_idx % n_vecs);  // 0,8,16,24

        for (uint ii = 0; ii < uint(SG_BM); ii += bstride) {
            const uint row = bi + ii; // 0..63
            const uint gr = base_i + row;

            #pragma clang loop unroll(full)
            for (uint v = 0; v < vec_size; ++v) {
                const uint col = bj + v;
                const uint gk = k0 + col;
                float val = 0.0f;
                if (gr < m && gk < n) {
                    val = B0[gr * ds.ldB + gk] * D_tg[col];
                }
                As_tg[row * uint(SG_LDA) + col] = val;
            }
        }

        for (uint ii = 0; ii < uint(SG_BN); ii += bstride) {
            const uint row = bi + ii;
            const uint gr = base_j + row;

            #pragma clang loop unroll(full)
            for (uint v = 0; v < vec_size; ++v) {
                const uint col = bj + v;
                const uint gk = k0 + col;
                float val = 0.0f;
                if (gr < m && gk < n) {
                    val = B0[gr * ds.ldB + gk];
                }
                Bs_tg[row * uint(SG_LDB) + col] = val;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        op.mma(As_tg, Bs_tg);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    op.store_sub(C0, ds.ldC, base_i, base_j, m);
}

// -----------------------------------------
// 6) Debug helper: CSR -> dense (for small tests)
// -----------------------------------------

kernel void csr_to_dense_f32(
    device const float* values        [[buffer(0)]],
    device const uint*  col_indices   [[buffer(1)]],
    device const uint*  row_ptr       [[buffer(2)]],
    device float*       out           [[buffer(3)]],
    constant uint&      m             [[buffer(4)]],
    constant uint&      n             [[buffer(5)]],
    constant uint&      ld            [[buffer(6)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= m) return;

    // Zero row
    for (uint j = 0; j < n; j++) {
        out[row * ld + j] = 0.0f;
    }

    const uint start = row_ptr[row];
    const uint end   = row_ptr[row + 1];
    for (uint idx = start; idx < end; idx++) {
        const uint col = col_indices[idx];
        if (col < n) {
            out[row * ld + col] = values[idx];
        }
    }
}

// -----------------------------------------
// 7) Optional GPU assembly helpers (gather/scatter)
// -----------------------------------------
//
// These are intentionally generic.
// They are useful for assembling dense frontals from sparse input or adding a child contribution
// into a parent frontal.
//
// Index maps are in element indices (flattened), not bytes.
//
// IMPORTANT: scatter_add_f32 is only race-free if map[] has no duplicates for the call.
//
//   Example use:
//     // Add child contribution (dense) into parent frontal (dense):
//     // src[k] corresponds to dst[map[k]]
//     scatter_add_f32(src, dst, map, len)
//
//     // Gather arbitrary entries:
//     gather_f32(src, out, map, len)

kernel void gather_f32(
    device const float* src     [[buffer(0)]],
    device const uint*  map     [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      n       [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = src[map[gid]];
}

kernel void scatter_set_f32(
    device const float* src     [[buffer(0)]],
    device float*       dst     [[buffer(1)]],
    device const uint*  map     [[buffer(2)]],
    constant uint&      n       [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    dst[map[gid]] = src[gid];
}

// Non-atomic scatter-add (requires unique indices per call)
kernel void scatter_add_f32(
    device const float* src     [[buffer(0)]],
    device float*       dst     [[buffer(1)]],
    device const uint*  map     [[buffer(2)]],
    constant uint&      n       [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    const uint idx = map[gid];
    dst[idx] += src[gid];
}
