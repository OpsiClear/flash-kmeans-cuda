// flash_kmeans_cuda/csrc/assign/assign_sm80_dslab_kernel.cuh
//
// Ampere+ (sm_80) D-slab variant of the assign kernel. Holds one D-slab
// (max 128 features) in SMEM at a time, with the K-chunk inner loop
// iterating slabs and accumulating cross-products in registers across
// slabs. SMEM footprint is D-independent at SLAB_MAX=128, so every
// target D ∈ {192, 224, 256, 320, 384} fits the BN=128 BK=128 STAGES=2
// 8-warp tile that hits 202 TFLOPS at D=128 in the legacy kernel.
//
// Cloned from assign_sm80_kernel.cuh in Task 2; restructured in Task 5
// to do per-slab SMEM staging with cross-slab partial_cross carry.
//
// This file must be included AFTER assign_sm80_kernel.cuh in any
// translation unit that uses both. The anonymous-namespace device helpers
// (mma_atom, async_load_tile, async_load_csq_full_tile, store_csq_tile)
// are defined once in assign_sm80_kernel.cuh; this file reuses them
// without re-declaring them.
//
// Include guards handled by #pragma once. Include this file only from .cu
// translation units that are compiled by nvcc.
#pragma once

#include "assign_kernel_launch.h"
#include "assign_common.cuh"
#include "../common/arch.cuh"
#include "../common/ptx.cuh"
// Pull in the anonymous-namespace helper device functions (mma_atom,
// async_load_tile, async_load_csq_full_tile, store_csq_tile).  #pragma once
// in assign_sm80_kernel.cuh makes this a no-op when both files land in
// the same TU; in a standalone TU (e.g., assign_sm80_dslab.cu in Task 3)
// the helpers are compiled exactly once.
#include "assign_sm80_kernel.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace fkc {
namespace assign {

namespace dslab_detail {

// Load one D-slab of a (rows_max × slab_w) fp16/bf16 tile from gmem into SMEM.
// Differs from async_load_tile in that the source row stride is D_FULL (the
// full row width) while only slab_w columns are copied. Without this, a slab
// load would have to assume rows_packed-at-slab_w in gmem, which isn't how
// our (B, K, D_FULL) tensors are laid out.
//
// Each thread copies 8 elts (16 bytes) per pass; loop over passes until the
// rows_max × slab_w region is covered.
template <typename T, int THREADS>
__device__ __forceinline__ void async_load_slab_tile(
    T* smem_tile,                 // (rows_max * D_SMEM_slab) destination base
    const T* gmem_tile,           // gmem row 0 of the slab (already + slab_off)
    int rows,                     // valid row count (<= rows_max)
    int rows_max,                 // SMEM row capacity
    int slab_w,                   // copy width (= one slab's feature count)
    int D_FULL_stride,            // gmem row stride (full row width)
    int D_SMEM_slab) {            // SMEM row stride (slab_w + pad)
  const int tid = threadIdx.x;
  const int total_elts = rows_max * slab_w;
  const int elts_per_load = 16 / sizeof(T);   // 8 for fp16/bf16
  for (int off = tid * elts_per_load; off < total_elts;
       off += THREADS * elts_per_load) {
    int row = off / slab_w;
    int col = off % slab_w;
    bool valid = row < rows;
    T* dst = smem_tile + (size_t)row * D_SMEM_slab + col;
    const T* src = gmem_tile + (size_t)row * D_FULL_stride + col;
    unsigned int dst_smem = ptx::cvta_to_shared(dst);
    ptx::cp_async_16B(dst_smem, src, valid);
  }
}

}  // namespace dslab_detail

// assign_sm80_dslab_kernel is defined at fkc::assign scope (not anonymous) so that
// the forward declaration in assign_dslab_variants.h can reference it.
//
// IMPORTANT: signature, __launch_bounds__, and the __restrict__ qualifiers on
// every parameter must stay in sync with the forward declaration in
// assign_dslab_variants.h. NVCC bakes __launch_bounds__ into the kernel symbol;
// a mismatch causes cudaErrorInvalidDeviceFunction at runtime, not at link
// time.
template <typename T,
          int BLOCK_N, int BLOCK_K, int WARPS_PER_CTA, int PIPE_STAGES,
          int N_TILES_PER_CTA,
          int D_FULL,
          int S0, int S1, int S2, int S3,
          int SMEM_PAD_SLAB,
          bool RAW_DIST>
__global__ void __launch_bounds__(WARPS_PER_CTA * 32, 1)
assign_sm80_dslab_kernel(
    const T* __restrict__ x,            // (B, N, D)
    const T* __restrict__ centroids,    // (B, K, D)
    const float* __restrict__ x_sq,     // (B, N) — unused when RAW_DIST
    const float* __restrict__ c_sq,     // (B, K)
    int32_t* __restrict__ cluster_ids,  // (B, N)
    int B, int N, int K, int D) {
  constexpr int THREADS_PER_CTA = WARPS_PER_CTA * 32;
  constexpr int WARP_M = BLOCK_N / WARPS_PER_CTA;
  constexpr int M_ATOMS_PER_WARP = WARP_M / 16;
  constexpr int N_ATOMS_PER_WARP = BLOCK_K / 8;
  static_assert(WARP_M >= 16 && (WARP_M % 16) == 0, "WARP_M must be a multiple of 16");
  static_assert((BLOCK_K % 8) == 0, "BLOCK_K must be a multiple of 8");
  static_assert(N_TILES_PER_CTA >= 1, "N_TILES_PER_CTA must be >= 1");
  static_assert(D_FULL > 0, "D-slab kernel requires D_FULL > 0");

  // Compile-time slab partition. S0..S3 are the per-slab feature widths; a
  // zero entry terminates the partition. Slab widths must sum to D_FULL.
  // (The {S0,S1,S2,S3} array literal is reconstructed inside the slab loop
  // so the compiler can index it as a constexpr.)
  constexpr int kNumSlabs =
      (S3 > 0) ? 4 :
      (S2 > 0) ? 3 :
      (S1 > 0) ? 2 :
      (S0 > 0) ? 1 : 0;
  static_assert(kNumSlabs > 0, "D-slab kernel requires non-empty partition");
  static_assert((S0 <= SLAB_MAX) && (S1 <= SLAB_MAX) &&
                (S2 <= SLAB_MAX) && (S3 <= SLAB_MAX),
                "every slab must be <= SLAB_MAX");
  static_assert((S0 + S1 + S2 + S3) == D_FULL,
                "slab widths must sum to D_FULL");
  static_assert((S0 % BLOCK_D) == 0 && (S1 % BLOCK_D == 0 || S1 == 0) &&
                (S2 % BLOCK_D == 0 || S2 == 0) &&
                (S3 % BLOCK_D == 0 || S3 == 0),
                "slab widths must be a multiple of BLOCK_D=16");

  constexpr int D_SLAB_SMEM = SLAB_MAX + SMEM_PAD_SLAB;

  // CTA-wide invariants (hoisted above the n_tile loop).
  const int pid_b = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarp;
  const int lane = tid % kWarp;

  // SMEM layout (one slab at a time, D-independent footprint):
  //   x_slab_smem [BLOCK_N * D_SLAB_SMEM]                  (re-loaded per slab)
  //   c_slab_smem [PIPE_STAGES * BLOCK_K * D_SLAB_SMEM]    (rotated per K-chunk)
  //   c_sq_smem   [PIPE_STAGES * BLOCK_K]
  // (x_sq lives in registers — see xs_top_cache / xs_bot_cache below.)
  extern __shared__ unsigned char smem_raw[];
  T* x_slab_smem = reinterpret_cast<T*>(smem_raw);
  T* c_slab_smem = x_slab_smem + (size_t)BLOCK_N * D_SLAB_SMEM;
  float* c_sq_smem = reinterpret_cast<float*>(
      c_slab_smem + (size_t)PIPE_STAGES * BLOCK_K * D_SLAB_SMEM);

  // Per-thread row/col indices used by both operand-load and epilogue.
  // mma m16n8k16 D distribution: per atom the thread holds 4 fp32 regs at
  // (M=lane/4, N=2*(lane%4)), (M=lane/4, N=2*(lane%4)+1),
  // (M=lane/4+8, N=2*(lane%4)), (M=lane/4+8, N=2*(lane%4)+1).
  const int row_top_in_warp = lane / 4;             // 0..7  (within 16-row atom)
  const int row_bot_in_warp = row_top_in_warp + 8;  // 8..15
  const int col_in_atom = (lane % 4) * 2;           // 0,2,4,6 within an 8-wide atom

  // ldmatrix lane-mapping constants (depend on lane only, not d_off / chunk).
  const int ldm_row_off    = (lane & 8)  ? 8 : 0;   // bit 3 -> M-half (A) / N-half (B)
  const int ldm_col_off    = (lane & 16) ? 8 : 0;   // bit 4 -> K-half
  const int ldm_row_in_half = lane & 7;
  const int ldm_n_atom_off = (lane & 8) ? 8 : 0;    // for B: bit 3 -> atom n vs n+1

  const int num_k_chunks = (K + BLOCK_K - 1) / BLOCK_K;

  // ====== Outer loop over N-tiles (persistent kernel) ======
  // Each CTA processes N_TILES_PER_CTA contiguous BLOCK_N rows of N. Per
  // n_tile we run the K-chunk loop (each K-chunk loops over D-slabs) +
  // write cluster_ids. cp.async pipeline state is fully drained between
  // tiles so the in-flight group counter stays predictable.
  #pragma unroll 1
  for (int n_tile = 0; n_tile < N_TILES_PER_CTA; ++n_tile) {
    const int n_start = (blockIdx.x * N_TILES_PER_CTA + n_tile) * BLOCK_N;
    const int n_count = min(BLOCK_N, N - n_start);
    if (n_count <= 0) break;  // tail CTA: remaining tiles are out of range

    // Per-tile state: best[], xs_*_cache, top/bot validity. RESET on every
    // n_tile iteration — caching them across tiles would corrupt results.
    Best best[M_ATOMS_PER_WARP * 2];
    #pragma unroll
    for (int i = 0; i < M_ATOMS_PER_WARP * 2; ++i) {
      best[i] = Best{FLT_MAX, 0};
    }

    // Cache per-thread x_sq values + row validity in registers. x_sq is per-row,
    // not per-slab, so it's loaded once per n_tile. Each thread owns
    // M_ATOMS_PER_WARP * 2 distinct rows. Loading once here avoids num_k_chunks
    // SMEM reads per row in the epilogue.
    float xs_top_cache[M_ATOMS_PER_WARP];
    float xs_bot_cache[M_ATOMS_PER_WARP];
    bool top_valid_cache[M_ATOMS_PER_WARP];
    bool bot_valid_cache[M_ATOMS_PER_WARP];
    #pragma unroll
    for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
      int row_top = warp_id * WARP_M + m * 16 + row_top_in_warp;
      int row_bot = warp_id * WARP_M + m * 16 + row_bot_in_warp;
      top_valid_cache[m] = row_top < n_count;
      bot_valid_cache[m] = row_bot < n_count;
      if constexpr (RAW_DIST) {
        // x_sq is constant across all candidate centroids for a row, so the
        // raw-distance argmin can omit it and skip these global loads.
        xs_top_cache[m] = 0.f;
        xs_bot_cache[m] = 0.f;
      } else {
        xs_top_cache[m] = top_valid_cache[m]
            ? x_sq[(size_t)pid_b * N + n_start + row_top] : 0.f;
        xs_bot_cache[m] = bot_valid_cache[m]
            ? x_sq[(size_t)pid_b * N + n_start + row_bot] : 0.f;
      }
    }

    // Helpers issuing per-slab cp.async loads. Closures capture the
    // n_tile-scope state (n_start, n_count) so the slab loop body stays
    // compact.
    auto issue_x_slab = [&](int slab_off, int slab_w) {
      dslab_detail::async_load_slab_tile<T, THREADS_PER_CTA>(x_slab_smem,
                         x + (size_t)pid_b * N * D_FULL + (size_t)n_start * D_FULL + slab_off,
                         n_count, BLOCK_N, slab_w, D_FULL, D_SLAB_SMEM);
    };
    auto issue_c_slab = [&](int chunk_idx, int stage, int slab_off, int slab_w) {
      int k_start = chunk_idx * BLOCK_K;
      int k_count = min(BLOCK_K, K - k_start);
      T* c_dst = c_slab_smem + (size_t)stage * BLOCK_K * D_SLAB_SMEM;
      dslab_detail::async_load_slab_tile<T, THREADS_PER_CTA>(c_dst,
                         centroids + (size_t)pid_b * K * D_FULL + (size_t)k_start * D_FULL + slab_off,
                         k_count, BLOCK_K, slab_w, D_FULL, D_SLAB_SMEM);
    };
    auto issue_csq = [&](int chunk_idx, int stage) {
      int k_start = chunk_idx * BLOCK_K;
      int k_count = min(BLOCK_K, K - k_start);
      float* csq_dst = c_sq_smem + stage * BLOCK_K;
      static_assert((BLOCK_K % 4) == 0, "BLOCK_K must be a multiple of 4 for csq copy");
      const float* csq_src = c_sq + (size_t)pid_b * K + k_start;
      // ASYNC_CSQ is hard-coded to true for the dslab kernel (K >= 256 use
      // case always benefits from async c_sq loads).
      if (k_count == BLOCK_K) {
        async_load_csq_full_tile<THREADS_PER_CTA>(csq_dst, csq_src, BLOCK_K);
      } else {
        store_csq_tile<THREADS_PER_CTA>(csq_dst, csq_src, k_count, BLOCK_K);
      }
    };

    for (int chunk_idx = 0; chunk_idx < num_k_chunks; ++chunk_idx) {
      int k_start = chunk_idx * BLOCK_K;
      int k_count = min(BLOCK_K, K - k_start);
      int stage = chunk_idx % PIPE_STAGES;
      T* c_tile_base = c_slab_smem + (size_t)stage * BLOCK_K * D_SLAB_SMEM;
      float* c_sq_tile = c_sq_smem + stage * BLOCK_K;

      // Per-warp fp32 accumulator for cross products, persistent across the
      // slab loop and zero-initialized at K-chunk start. Each (m, n) atom owns
      // 4 fp32 values: top-row col0, top-row col1, bot-row col0, bot-row col1.
      // (Matches the legacy fp16-acc layout after unpacking.)
      float partial_cross[M_ATOMS_PER_WARP][N_ATOMS_PER_WARP][4];
      #pragma unroll
      for (int m = 0; m < M_ATOMS_PER_WARP; ++m)
        #pragma unroll
        for (int n = 0; n < N_ATOMS_PER_WARP; ++n) {
          partial_cross[m][n][0] = 0.f;
          partial_cross[m][n][1] = 0.f;
          partial_cross[m][n][2] = 0.f;
          partial_cross[m][n][3] = 0.f;
        }

      // ----- Slab loop: stage one slab into SMEM, mma, accumulate, repeat -----
      // All slabs within a K-chunk share the same SMEM stage (PIPE_STAGES
      // rotates per K-chunk, not per slab). cp.async wait_all between slabs
      // is mandatory because successive slabs overwrite the same SMEM. This
      // is the per-slab sync overhead the design accepts; a follow-up could
      // pipeline with extra SMEM slots if benchmarks justify it.
      int slab_off = 0;
      #pragma unroll
      for (int slab_idx = 0; slab_idx < kNumSlabs; ++slab_idx) {
        constexpr int slab_w_arr[MAX_SLABS] = {S0, S1, S2, S3};
        const int slab_w = slab_w_arr[slab_idx];

        // Issue async loads for x slab + c slab. c_sq is per K-chunk, so issue
        // it only on slab 0 (alongside the first c slab).
        issue_x_slab(slab_off, slab_w);
        issue_c_slab(chunk_idx, stage, slab_off, slab_w);
        if (slab_idx == 0) {
          issue_csq(chunk_idx, stage);
        }
        ptx::cp_async_commit();
        // PIPE_STAGES > 1 is currently STRUCTURALLY INERT in the dslab kernel:
        // cp_async_wait_all() here drains every outstanding cp.async group, so
        // c_slab_smem only ever needs one stage's worth of SMEM. Catalog entries
        // should pass STAGES=1 to avoid wasting ~32 KB of SMEM per CTA at
        // BK=128. A future change could replace wait_all with
        // cp_async_wait_group<0> and overlap slab N+1's load with slab N's
        // mma — that would unlock real PIPE_STAGES > 1 benefit, but is out of
        // scope for this kernel.
        ptx::cp_async_wait_all();
        __syncthreads();

        // Per-warp accumulator for THIS slab only: fp16 packed regs (matches
        // legacy mma_atom path). Unpacked into partial_cross[][] at end of slab.
        uint32_t acc[M_ATOMS_PER_WARP][N_ATOMS_PER_WARP][2];
        #pragma unroll
        for (int m = 0; m < M_ATOMS_PER_WARP; ++m)
          #pragma unroll
          for (int n = 0; n < N_ATOMS_PER_WARP; ++n) {
            acc[m][n][0] = 0u;
            acc[m][n][1] = 0u;
          }

        // Walk this slab in BLOCK_D=16 steps; each step does one mma per
        // (m, n) atom. Unroll up to 8 iters (covers slab_w=128); for narrower
        // slabs the compiler unrolls fully.
        #pragma unroll 8
        for (int d_off = 0; d_off < slab_w; d_off += BLOCK_D) {
          // ----- Load A regs via ldmatrix.x4 -----
          uint32_t a_regs[M_ATOMS_PER_WARP][4];
          #pragma unroll
          for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
            int m_base = warp_id * WARP_M + m * 16;
            int row = m_base + ldm_row_off + ldm_row_in_half;
            unsigned int smem_addr = ptx::cvta_to_shared(
                x_slab_smem + (size_t)row * D_SLAB_SMEM + d_off + ldm_col_off);
            ptx::ldmatrix_x4(a_regs[m][0], a_regs[m][1],
                             a_regs[m][2], a_regs[m][3], smem_addr);
          }

          // ----- Load B regs via ldmatrix.x4 (no trans), 2 atoms at a time -----
          uint32_t b_regs[N_ATOMS_PER_WARP][2];
          static_assert((N_ATOMS_PER_WARP % 2) == 0,
                        "ldmatrix.x4 covers 2 N-atoms; N_ATOMS must be even");
          #pragma unroll
          for (int n = 0; n < N_ATOMS_PER_WARP; n += 2) {
            int n_col = n * 8 + ldm_n_atom_off + ldm_row_in_half;
            unsigned int smem_addr = ptx::cvta_to_shared(
                c_tile_base + (size_t)n_col * D_SLAB_SMEM + d_off + ldm_col_off);
            uint32_t r0, r1, r2, r3;
            ptx::ldmatrix_x4(r0, r1, r2, r3, smem_addr);
            b_regs[n][0]     = r0;     // atom n,   K-half 0 (b0)
            b_regs[n + 1][0] = r1;     // atom n+1, K-half 0 (b0)
            b_regs[n][1]     = r2;     // atom n,   K-half 1 (b1)
            b_regs[n + 1][1] = r3;     // atom n+1, K-half 1 (b1)
          }

          // ----- Issue mma atoms (fp16 acc, accumulating within this slab) -----
          #pragma unroll
          for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
            #pragma unroll
            for (int n = 0; n < N_ATOMS_PER_WARP; ++n) {
              mma_atom<T>(
                  acc[m][n][0], acc[m][n][1],
                  a_regs[m][0], a_regs[m][1], a_regs[m][2], a_regs[m][3],
                  b_regs[n][0], b_regs[n][1],
                  acc[m][n][0], acc[m][n][1]);
            }
          }
        }  // d-loop within slab

        // Unpack this slab's fp16 acc and add into fp32 partial_cross. The
        // fp16->fp32 conversion + add is exact for typical magnitudes; we
        // pay 4 fmas per atom per slab, negligible vs the mma cost.
        #pragma unroll
        for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
          #pragma unroll
          for (int n = 0; n < N_ATOMS_PER_WARP; ++n) {
            __half2 packed_top = *reinterpret_cast<const __half2*>(&acc[m][n][0]);
            __half2 packed_bot = *reinterpret_cast<const __half2*>(&acc[m][n][1]);
            partial_cross[m][n][0] += __low2float(packed_top);
            partial_cross[m][n][1] += __high2float(packed_top);
            partial_cross[m][n][2] += __low2float(packed_bot);
            partial_cross[m][n][3] += __high2float(packed_bot);
          }
        }

        slab_off += slab_w;
      }  // slab loop

      // ----- In-register epilogue: convert cross-product to distance and reduce -----
      if (k_count == BLOCK_K) {
        // Hot path: all K columns in this chunk are valid.
        if (n_count == BLOCK_N) {
          #pragma unroll
          for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
            [[maybe_unused]] const float xs_top = xs_top_cache[m];
            [[maybe_unused]] const float xs_bot = xs_bot_cache[m];

            #pragma unroll
            for (int n = 0; n < N_ATOMS_PER_WARP; ++n) {
              int k_in_chunk_0 = n * 8 + col_in_atom;
              int k_in_chunk_1 = k_in_chunk_0 + 1;
              int k_global_0 = k_start + k_in_chunk_0;
              int k_global_1 = k_start + k_in_chunk_1;
              float cs0 = c_sq_tile[k_in_chunk_0];
              float cs1 = c_sq_tile[k_in_chunk_1];

              float a_top0 = partial_cross[m][n][0];
              float a_top1 = partial_cross[m][n][1];
              float a_bot0 = partial_cross[m][n][2];
              float a_bot1 = partial_cross[m][n][3];

              if constexpr (RAW_DIST) {
                update_best(best[m * 2 + 0], cs0 - 2.0f * a_top0, k_global_0);
                update_best(best[m * 2 + 0], cs1 - 2.0f * a_top1, k_global_1);
                update_best(best[m * 2 + 1], cs0 - 2.0f * a_bot0, k_global_0);
                update_best(best[m * 2 + 1], cs1 - 2.0f * a_bot1, k_global_1);
              } else {
                update_best(best[m * 2 + 0], to_dist(a_top0, xs_top, cs0), k_global_0);
                update_best(best[m * 2 + 0], to_dist(a_top1, xs_top, cs1), k_global_1);
                update_best(best[m * 2 + 1], to_dist(a_bot0, xs_bot, cs0), k_global_0);
                update_best(best[m * 2 + 1], to_dist(a_bot1, xs_bot, cs1), k_global_1);
              }
            }
          }
        } else {
          #pragma unroll
          for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
            const bool top_valid = top_valid_cache[m];
            const bool bot_valid = bot_valid_cache[m];
            [[maybe_unused]] const float xs_top = xs_top_cache[m];
            [[maybe_unused]] const float xs_bot = xs_bot_cache[m];

            #pragma unroll
            for (int n = 0; n < N_ATOMS_PER_WARP; ++n) {
              int k_in_chunk_0 = n * 8 + col_in_atom;
              int k_in_chunk_1 = k_in_chunk_0 + 1;
              int k_global_0 = k_start + k_in_chunk_0;
              int k_global_1 = k_start + k_in_chunk_1;
              float cs0 = c_sq_tile[k_in_chunk_0];
              float cs1 = c_sq_tile[k_in_chunk_1];

              float a_top0 = partial_cross[m][n][0];
              float a_top1 = partial_cross[m][n][1];
              float a_bot0 = partial_cross[m][n][2];
              float a_bot1 = partial_cross[m][n][3];

              if (top_valid) {
                if constexpr (RAW_DIST) {
                  update_best(best[m * 2 + 0], cs0 - 2.0f * a_top0, k_global_0);
                  update_best(best[m * 2 + 0], cs1 - 2.0f * a_top1, k_global_1);
                } else {
                  update_best(best[m * 2 + 0], to_dist(a_top0, xs_top, cs0), k_global_0);
                  update_best(best[m * 2 + 0], to_dist(a_top1, xs_top, cs1), k_global_1);
                }
              }
              if (bot_valid) {
                if constexpr (RAW_DIST) {
                  update_best(best[m * 2 + 1], cs0 - 2.0f * a_bot0, k_global_0);
                  update_best(best[m * 2 + 1], cs1 - 2.0f * a_bot1, k_global_1);
                } else {
                  update_best(best[m * 2 + 1], to_dist(a_bot0, xs_bot, cs0), k_global_0);
                  update_best(best[m * 2 + 1], to_dist(a_bot1, xs_bot, cs1), k_global_1);
                }
              }
            }
          }
        }
      } else {
        #pragma unroll
        for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
          const bool top_valid = top_valid_cache[m];
          const bool bot_valid = bot_valid_cache[m];
          [[maybe_unused]] const float xs_top = xs_top_cache[m];
          [[maybe_unused]] const float xs_bot = xs_bot_cache[m];

          #pragma unroll
          for (int n = 0; n < N_ATOMS_PER_WARP; ++n) {
            int k_in_chunk_0 = n * 8 + col_in_atom;
            int k_in_chunk_1 = k_in_chunk_0 + 1;
            int k_global_0 = k_start + k_in_chunk_0;
            int k_global_1 = k_start + k_in_chunk_1;
            bool k0_valid = k_global_0 < K;
            bool k1_valid = k_global_1 < K;
            float cs0 = c_sq_tile[k_in_chunk_0];
            float cs1 = c_sq_tile[k_in_chunk_1];

            float a_top0 = partial_cross[m][n][0];
            float a_top1 = partial_cross[m][n][1];
            float a_bot0 = partial_cross[m][n][2];
            float a_bot1 = partial_cross[m][n][3];

            if (top_valid) {
              if (k0_valid) {
                if constexpr (RAW_DIST) {
                  update_best(best[m * 2 + 0], cs0 - 2.0f * a_top0, k_global_0);
                } else {
                  update_best(best[m * 2 + 0], to_dist(a_top0, xs_top, cs0), k_global_0);
                }
              }
              if (k1_valid) {
                if constexpr (RAW_DIST) {
                  update_best(best[m * 2 + 0], cs1 - 2.0f * a_top1, k_global_1);
                } else {
                  update_best(best[m * 2 + 0], to_dist(a_top1, xs_top, cs1), k_global_1);
                }
              }
            }
            if (bot_valid) {
              if (k0_valid) {
                if constexpr (RAW_DIST) {
                  update_best(best[m * 2 + 1], cs0 - 2.0f * a_bot0, k_global_0);
                } else {
                  update_best(best[m * 2 + 1], to_dist(a_bot0, xs_bot, cs0), k_global_0);
                }
              }
              if (k1_valid) {
                if constexpr (RAW_DIST) {
                  update_best(best[m * 2 + 1], cs1 - 2.0f * a_bot1, k_global_1);
                } else {
                  update_best(best[m * 2 + 1], to_dist(a_bot1, xs_bot, cs1), k_global_1);
                }
              }
            }
          }
        }
      }
    }  // k-chunk loop

    // ----- Reduce best[] across the 4 lanes that share a row within one atom -----
    // Within a 4-lane sub-group sharing lane/4, the 4 lanes hold candidate K
    // cols 0,2,4,6 + their +1 (already folded into best[] above). __shfl_xor
    // reduces by xoring the lane index within the 4-lane group.
    auto warp_reduce_row = [&](Best& b) {
      #pragma unroll
      for (int offset : {1, 2}) {
        float other_d = __shfl_xor_sync(0xffffffff, b.dist, offset, 4);
        int   other_i = __shfl_xor_sync(0xffffffff, b.idx,  offset, 4);
        // Match update_best's tie-break: lower distance wins; on tie keep
        // the smaller idx (here, the one already held when other_d == b.dist).
        if (other_d < b.dist || (other_d == b.dist && other_i < b.idx)) {
          b.dist = other_d;
          b.idx = other_i;
        }
      }
    };

    #pragma unroll
    for (int i = 0; i < M_ATOMS_PER_WARP * 2; ++i) {
      warp_reduce_row(best[i]);
    }

    // Lane (lane % 4 == 0) holds the row's answer; write cluster_ids.
    if ((lane % 4) == 0) {
      #pragma unroll
      for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
        int row_top = warp_id * WARP_M + m * 16 + row_top_in_warp;
        int row_bot = warp_id * WARP_M + m * 16 + row_bot_in_warp;
        if (row_top < n_count) {
          cluster_ids[(size_t)pid_b * N + n_start + row_top] = best[m * 2 + 0].idx;
        }
        if (row_bot < n_count) {
          cluster_ids[(size_t)pid_b * N + n_start + row_bot] = best[m * 2 + 1].idx;
        }
      }
    }

    // Drain ALL cp.async groups before next n_tile. Each slab inside the
    // K-chunk loop already does its own wait_all, so by the time the K-chunk
    // loop ends nothing should be in flight — but issue the drain
    // defensively (matches the legacy pattern).
    if (N_TILES_PER_CTA > 1) {
      ptx::cp_async_wait_all();
      __syncthreads();
    }
  }  // n_tile loop
}

}  // namespace assign
}  // namespace fkc
