// flash_kmeans_cuda/csrc/assign/assign_sm80_dslab_kernel.cuh
//
// Ampere+ (sm_80) D-slab variant of the assign kernel. Holds one D-slab
// (max 128 features) in SMEM at a time, with the K-chunk inner loop
// iterating slabs and accumulating cross-products in registers across
// slabs. SMEM footprint is D-independent at SLAB_MAX=128, so every
// target D ∈ {192, 224, 256, 320, 384} fits the BN=128 BK=128 STAGES=2
// 8-warp tile that hits 202 TFLOPS at D=128 in the legacy kernel.
//
// Cloned from assign_sm80_kernel.cuh in Task 2 of the D-slab plan; the
// inner loop is restructured in Task 4. Until Task 5, this kernel is
// behaviorally identical to the legacy at supported D values.
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
  // S0..S3 partition + SMEM_PAD_SLAB + slab loop are wired in Task 5.
  // This task is scaffolding — the kernel body still uses D_FULL as a
  // single-shot D dimension (legacy-equivalent behavior).
  (void)S0; (void)S1; (void)S2; (void)S3;
  (void)SMEM_PAD_SLAB;

  // CTA-wide invariants (hoisted above the n_tile loop).
  const int pid_b = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarp;
  const int lane = tid % kWarp;
  const int D_TILE = D_FULL;
  const int D_SMEM = D_TILE + SMEM_PAD;

  // SMEM layout (with row-stride padding D_SMEM = D + SMEM_PAD):
  //   x_smem [BLOCK_N * D_SMEM]              (re-loaded per n_tile)
  //   c_smem [PIPE_STAGES * BLOCK_K * D_SMEM]  (rotated within a tile)
  //   c_sq_smem [PIPE_STAGES * BLOCK_K]
  // (x_sq lives in registers — see xs_top_cache / xs_bot_cache below.)
  extern __shared__ unsigned char smem_raw[];
  T* x_smem = reinterpret_cast<T*>(smem_raw);
  T* c_smem = x_smem + (size_t)BLOCK_N * D_SMEM;
  float* c_sq_smem = reinterpret_cast<float*>(
      c_smem + (size_t)PIPE_STAGES * BLOCK_K * D_SMEM);

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
  // n_tile we fully load + consume its x_smem + run the K-chunk loop +
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

    // Load x_tile (CTA-wide). For tile 0 we load fresh; for subsequent
    // tiles the load was already issued by the previous tile's epilogue
    // and made visible by the inter-tile cp_async_wait_all + __syncthreads,
    // so we skip — saves the latency of an extra cp.async + sync pair.
    if (n_tile == 0) {
      async_load_tile<T, THREADS_PER_CTA>(x_smem,
                         x + (size_t)pid_b * N * D_TILE + (size_t)n_start * D_TILE,
                         n_count, BLOCK_N, D_TILE, D_SMEM);
      ptx::cp_async_commit();
    }

    auto issue_c_chunk = [&](int chunk_idx, int stage) {
      int k_start = chunk_idx * BLOCK_K;
      int k_count = min(BLOCK_K, K - k_start);
      T* c_dst = c_smem + (size_t)stage * BLOCK_K * D_SMEM;
      float* csq_dst = c_sq_smem + stage * BLOCK_K;
      async_load_tile<T, THREADS_PER_CTA>(c_dst,
                         centroids + (size_t)pid_b * K * D_TILE + (size_t)k_start * D_TILE,
                         k_count, BLOCK_K, D_TILE, D_SMEM);
      static_assert((BLOCK_K % 4) == 0, "BLOCK_K must be a multiple of 4 for csq copy");
      const float* csq_src = c_sq + (size_t)pid_b * K + k_start;
      // ASYNC_CSQ is hard-coded to true for the dslab kernel (K >= 256 use
      // case always benefits from async c_sq loads).
      if (k_count == BLOCK_K) {
        async_load_csq_full_tile<THREADS_PER_CTA>(csq_dst, csq_src, BLOCK_K);
      } else {
        store_csq_tile<THREADS_PER_CTA>(csq_dst, csq_src, k_count, BLOCK_K);
      }
      ptx::cp_async_commit();
    };

    // Prime the pipeline: pre-issue up to PIPE_STAGES initial K-chunks.
    #pragma unroll
    for (int s = 0; s < PIPE_STAGES; ++s) {
      if (s < num_k_chunks) issue_c_chunk(s, s);
    }

    // Wait for x_tile + first c chunk (i.e., everything but the last
    // outstanding group). x_smem is only in pending for tile 0; for n>0
    // it was drained in the prior tile's wait_all so we just wait for c.
    ptx::cp_async_wait_group<PIPE_STAGES - 1>();
    __syncthreads();

    // Cache per-thread x_sq values + row validity in registers. Each thread
    // owns M_ATOMS_PER_WARP * 2 distinct rows. Loading once here avoids
    // num_k_chunks SMEM reads per row in the epilogue. NOTE: hoisting these
    // earlier (before the priming) regressed ~2% on mega — the gmem reads
    // compete with cp.async issues for memory subsystem bandwidth. Keep
    // them after the wait+sync.
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

    for (int chunk_idx = 0; chunk_idx < num_k_chunks; ++chunk_idx) {
    int k_start = chunk_idx * BLOCK_K;
    int k_count = min(BLOCK_K, K - k_start);
    int stage = chunk_idx % PIPE_STAGES;
    T* c_tile = c_smem + (size_t)stage * BLOCK_K * D_SMEM;
    float* c_sq_tile = c_sq_smem + stage * BLOCK_K;

    // Per-warp accumulator: [M_ATOMS][N_ATOMS][2] packed fp16 regs/thread.
    // Each u32 reg holds 2 fp16 values: d0 = (top-row col0, top-row col1),
    // d1 = (bot-row col0, bot-row col1).
    uint32_t acc[M_ATOMS_PER_WARP][N_ATOMS_PER_WARP][2];
    #pragma unroll
    for (int m = 0; m < M_ATOMS_PER_WARP; ++m)
      #pragma unroll
      for (int n = 0; n < N_ATOMS_PER_WARP; ++n) {
        acc[m][n][0] = 0u;
        acc[m][n][1] = 0u;
      }

    // Walk D in BLOCK_D=16 steps; each step does one mma per (m, n) atom.
    // Unroll up to 8 iters (covers D=64/128); for larger D the compiler
    // partially unrolls. Unrolling lets nvcc software-pipeline the
    // next-iter ldmatrix loads with the current-iter mma issues, hiding
    // mma's ~16-cycle latency.
    #pragma unroll 8
    for (int d_off = 0; d_off < D_TILE; d_off += BLOCK_D) {
      // ----- Load A regs via ldmatrix.x4 -----
      // Source: x_smem (BLOCK_N, D_SMEM) row-major fp16. Per m-atom, load a
      // 16x16 sub-tile starting at (m_base, d_off). The 4 8x8 sub-matrices
      // map to mma A's 4 regs in order (M-half × K-half).
      uint32_t a_regs[M_ATOMS_PER_WARP][4];
      #pragma unroll
      for (int m = 0; m < M_ATOMS_PER_WARP; ++m) {
        int m_base = warp_id * WARP_M + m * 16;
        int row = m_base + ldm_row_off + ldm_row_in_half;
        unsigned int smem_addr = ptx::cvta_to_shared(
            x_smem + (size_t)row * D_SMEM + d_off + ldm_col_off);
        ptx::ldmatrix_x4(a_regs[m][0], a_regs[m][1],
                         a_regs[m][2], a_regs[m][3], smem_addr);
      }

      // ----- Load B regs via ldmatrix.x4 (no trans), 2 atoms at a time -----
      // Source: c_tile (BLOCK_K, D_SMEM) row-major. From mma's perspective B
      // is (K=16, N=8) col-major, but our source memory has source-row = N
      // (centroid) and source-col = K (feature). ldmatrix.x4 (no trans) post-
      // load distribution: reg m of thread t = source[row=t/4, col=2*(t%4)..+1].
      // With (row=N, col=K), this gives reg = (N=t/4, K=2*(t%4)..+1) — which
      // is exactly mma B's b0 layout for one atom. The 4 sub-matrices in a
      // single ldmatrix.x4 cover 2 N-atoms × 2 K-halves, mapping to:
      //   matrix 0: N atom 0,  K-half 0  -> atom 0 b0
      //   matrix 1: N atom 1,  K-half 0  -> atom 1 b0  (N-half = bit 3)
      //   matrix 2: N atom 0,  K-half 1  -> atom 0 b1  (K-half = bit 4)
      //   matrix 3: N atom 1,  K-half 1  -> atom 1 b1
      uint32_t b_regs[N_ATOMS_PER_WARP][2];
      static_assert((N_ATOMS_PER_WARP % 2) == 0,
                    "ldmatrix.x4 covers 2 N-atoms; N_ATOMS must be even");
      #pragma unroll
      for (int n = 0; n < N_ATOMS_PER_WARP; n += 2) {
        // Lane addressing: bit 3 selects N-half (atoms n vs n+1 in this batch),
        // bit 4 selects K-half (d_off vs d_off+8). bits 0..2 select the row
        // within the 8-row matrix (= lane%8 → centroid offset within atom).
        int n_col = n * 8 + ldm_n_atom_off + ldm_row_in_half;
        unsigned int smem_addr = ptx::cvta_to_shared(
            c_tile + (size_t)n_col * D_SMEM + d_off + ldm_col_off);
        uint32_t r0, r1, r2, r3;
        ptx::ldmatrix_x4(r0, r1, r2, r3, smem_addr);
        b_regs[n][0]     = r0;     // atom n,   K-half 0 (b0)
        b_regs[n + 1][0] = r1;     // atom n+1, K-half 0 (b0)
        b_regs[n][1]     = r2;     // atom n,   K-half 1 (b1)
        b_regs[n + 1][1] = r3;     // atom n+1, K-half 1 (b1)
      }

      // ----- Issue mma atoms (fp16 acc) -----
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
    }  // d-loop

    // ----- In-register epilogue: convert cross-product to distance and reduce -----
    if (k_count == BLOCK_K) {
      // Hot path: all K columns in this chunk are valid. Avoid per-candidate
      // K-bound checks; only the final partial chunk needs them.
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

            __half2 packed_top = *reinterpret_cast<const __half2*>(&acc[m][n][0]);
            __half2 packed_bot = *reinterpret_cast<const __half2*>(&acc[m][n][1]);
            float a_top0 = __low2float(packed_top);
            float a_top1 = __high2float(packed_top);
            float a_bot0 = __low2float(packed_bot);
            float a_bot1 = __high2float(packed_bot);

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

            __half2 packed_top = *reinterpret_cast<const __half2*>(&acc[m][n][0]);
            __half2 packed_bot = *reinterpret_cast<const __half2*>(&acc[m][n][1]);
            float a_top0 = __low2float(packed_top);
            float a_top1 = __high2float(packed_top);
            float a_bot0 = __low2float(packed_bot);
            float a_bot1 = __high2float(packed_bot);

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

          __half2 packed_top = *reinterpret_cast<const __half2*>(&acc[m][n][0]);
          __half2 packed_bot = *reinterpret_cast<const __half2*>(&acc[m][n][1]);
          float a_top0 = __low2float(packed_top);
          float a_top1 = __high2float(packed_top);
          float a_bot0 = __low2float(packed_bot);
          float a_bot1 = __high2float(packed_bot);

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

    // Prefetch chunk_idx + PIPE_STAGES (if any) before waiting.
    int prefetch_idx = chunk_idx + PIPE_STAGES;
    if (prefetch_idx < num_k_chunks) {
      issue_c_chunk(prefetch_idx, prefetch_idx % PIPE_STAGES);
    }
    if (chunk_idx + 1 < num_k_chunks) {
      ptx::cp_async_wait_group<PIPE_STAGES - 1>();
      __syncthreads();
    }
  }  // k-chunk loop

  // ----- Inter-tile prefetch: issue NEXT tile's x_smem load now -----
  // x_smem is no longer read by this tile (K-loop done; reduce + cluster_ids
  // write below are register-only / output-only). Issuing the prefetch here
  // lets the cp.async overlap with the warp-shfl reductions and the
  // cluster_ids store, hiding the load latency from the next tile's
  // critical path. The drain below (wait_all + sync) makes it visible.
  //
  // We need a CTA sync before issuing because cp.async writes are async and
  // could land before all warps finish their last ldmatrix on the old
  // x_smem rows — without sync that's a write-during-read race.
  if (N_TILES_PER_CTA > 1 && (n_tile + 1) < N_TILES_PER_CTA) {
    int n_start_next = (blockIdx.x * N_TILES_PER_CTA + n_tile + 1) * BLOCK_N;
    int n_count_next = min(BLOCK_N, N - n_start_next);
    if (n_count_next > 0) {
      __syncthreads();
      async_load_tile<T, THREADS_PER_CTA>(x_smem,
                         x + (size_t)pid_b * N * D_TILE + (size_t)n_start_next * D_TILE,
                         n_count_next, BLOCK_N, D_TILE, D_SMEM);
      ptx::cp_async_commit();
    }
  }

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

  // Drain ALL cp.async groups before next n_tile. The K-chunk loop's
  // prefetch leaves up to PIPE_STAGES-1 trailing groups in flight, plus
  // this tile's x_smem' prefetch. Without draining, the next tile's
  // wait_group<PIPE_STAGES-1> drops the right count of groups but the HW
  // re-orders work in a way that ends up slower (~10% on mega) than
  // explicitly waiting here. Likely related to cp.async resource
  // accounting in the SM. __syncthreads() ensures all warps reach the
  // drain together (cp.async.wait_all is per-warp).
  if (N_TILES_PER_CTA > 1) {
    ptx::cp_async_wait_all();
    __syncthreads();
  }
  }  // n_tile loop
}

}  // namespace assign
}  // namespace fkc
