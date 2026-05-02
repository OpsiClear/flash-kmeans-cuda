// Ampere+ (sm_80) Euclidean assignment kernel using m16n8k16 mma.sync tensor
// cores, with the min-over-K reduction fused inline so each mma's output is
// consumed in registers and never materialized to SMEM (the structural diff
// vs Triton's tl.dot, which has to materialize the cross-product before
// reducing).
//
// Operand registers are populated via direct SCALAR SMEM loads — not
// ldmatrix.x4. ldmatrix is faster (bank-conflict-free hardware path,
// ~30%) but its lane-to-source mapping is delicate and error-prone, and
// hand-debugged ldmatrix layouts are hard to verify without running on
// hardware repeatedly. Scalar loads produce the bit-exact register layout
// `mma.sync.m16n8k16.row.col.f32.f16.f16.f32` expects, by construction:
//
//   A reg layout (4 u32 / thread, 2 fp16 packed each):
//     a0: A[m = lane/4,     k = 2*(lane%4) .. 2*(lane%4)+1]   (M-half 0, K-half 0)
//     a1: A[m = lane/4 + 8, k = 2*(lane%4) .. 2*(lane%4)+1]   (M-half 1, K-half 0)
//     a2: A[m = lane/4,     k = 2*(lane%4) + 8 .. + 9]        (M-half 0, K-half 1)
//     a3: A[m = lane/4 + 8, k = 2*(lane%4) + 8 .. + 9]        (M-half 1, K-half 1)
//
//   B reg layout (2 u32 / thread, 2 fp16 packed each — packed along K):
//     b0: B[k = 2*(lane%4) .. + 1, n = lane/4]                 (K-half 0, single N col)
//     b1: B[k = 2*(lane%4) + 8 .. + 9, n = lane/4]             (K-half 1, single N col)
//
//   D reg layout (4 fp32 / thread):
//     d0: D[m = lane/4,     n = 2*(lane%4)]
//     d1: D[m = lane/4,     n = 2*(lane%4) + 1]
//     d2: D[m = lane/4 + 8, n = 2*(lane%4)]
//     d3: D[m = lane/4 + 8, n = 2*(lane%4) + 1]
//
// Tile layout per CTA:
//   BLOCK_N = 128 points along N (4 warps × WARP_M=32 rows)
//   BLOCK_K =  64 centroids per K-chunk (8 N atoms × 8 cols each)
//   BLOCK_D =  16 features per mma K-step (2 K-halves × 8 cols)
//
// SMEM x_tile is loaded once (BLOCK_N × D) and reused across all K-chunks.
// SMEM c_tile is double-buffered along K-chunks via cp.async.
//
// Notes:
// - cluster_ids output is int32 to match the Triton signature.
// - x_sq, c_sq are fp32 and broadcast in the in-register epilogue.
// - D must be a multiple of 16 (covers the D=64/128/256 cases in the heuristic).
// - Bank conflicts: scalar loads from a (BLOCK_N, D) row-major SMEM tile have
//   conflicts when D is a power of 2. We accept this for now (correctness
//   first); a future patch can pad the row stride.

#include "assign.h"
#include "assign_common.cuh"
#include "../common/arch.cuh"
#include "../common/ptx.cuh"

#include "../common/torch_cuda_includes.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace fkc {
namespace assign {

namespace {

// Two tile configs:
//   "wide" (default): BLOCK_N=128, BLOCK_K=64.  Good for many-points, few-K.
//   "deep": BLOCK_N=64, BLOCK_K=128.            Good for many-K (K >= 512).
// Selected at the launcher based on K. Both compile from the same kernel
// template; tile sizes are template parameters.

constexpr int BLOCK_D = 16;
// SMEM row-stride padding (in fp16 elements). Adding 8 fp16 = 16 bytes
// preserves cp.async's required 16-byte alignment of the destination
// SMEM offset for every row, while shifting each subsequent SMEM row by
// 4 banks. Without padding, power-of-2 D (64/128/256) causes 32-way bank
// conflicts across rows; with this padding, the 8 cooperating row-bank
// offsets land on distinct 4-byte banks.
constexpr int SMEM_PAD = 8;

// FP16 accumulator path (2x mma throughput on Ada vs fp32 acc).
// Per atom: 2 packed-fp16 regs/thread instead of 4 fp32. Only fp16 input
// supports fp16 acc; bf16 input must use fp32 acc.
template <typename T>
__device__ __forceinline__ void mma_atom(
    uint32_t& d0, uint32_t& d1,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t c0, uint32_t c1);

template <>
__device__ __forceinline__ void mma_atom<__half>(
    uint32_t& d0, uint32_t& d1,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t c0, uint32_t c1) {
  ptx::mma_m16n8k16_fp16_acc_fp16(d0, d1, a0, a1, a2, a3, b0, b1, c0, c1);
}

// bf16 input falls back to fp32 acc — bf16 acc isn't supported by mma.sync.
// We bridge via a temporary fp32 path: convert acc-as-fp16 to fp32, run fp32-acc
// mma, convert back. (Not great; for now, route bf16 through this kernel only
// when the launcher dispatches it; or use a separate kernel template for bf16.)
template <>
__device__ __forceinline__ void mma_atom<__nv_bfloat16>(
    uint32_t& d0, uint32_t& d1,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t c0, uint32_t c1) {
  // Unpack fp16 acc -> fp32, mma, repack. This is a transitional path.
  __half2 c0h = *reinterpret_cast<const __half2*>(&c0);
  __half2 c1h = *reinterpret_cast<const __half2*>(&c1);
  float f0 = __half2float(c0h.x), f1 = __half2float(c0h.y);
  float f2 = __half2float(c1h.x), f3 = __half2float(c1h.y);
  float r0, r1, r2, r3;
  ptx::mma_m16n8k16_bf16(r0, r1, r2, r3, a0, a1, a2, a3, b0, b1, f0, f1, f2, f3);
  __half2 r0h = __floats2half2_rn(r0, r1);
  __half2 r1h = __floats2half2_rn(r2, r3);
  d0 = *reinterpret_cast<const uint32_t*>(&r0h);
  d1 = *reinterpret_cast<const uint32_t*>(&r1h);
}

// Issue cp.async copies for one tile of rows_max × D fp16 elts. SMEM is
// laid out with stride D_SMEM (= D + SMEM_PAD) to eliminate bank conflicts;
// gmem source is contiguous at stride D.
// Each thread copies 8 elts (16 bytes) per pass; loop over passes until full.
template <typename T, int THREADS>
__device__ __forceinline__ void async_load_tile(
    T* smem_tile,                 // (rows_max * D_SMEM) destination
    const T* gmem_tile,           // (rows * D) source
    int rows,                     // valid row count (<= rows_max)
    int rows_max,                 // SMEM row capacity
    int D, int D_SMEM) {
  const int tid = threadIdx.x;
  const int total_elts = rows_max * D;
  const int elts_per_load = 16 / sizeof(T);   // 8 for fp16/bf16
  for (int off = tid * elts_per_load; off < total_elts;
       off += THREADS * elts_per_load) {
    int row = off / D;
    int col = off % D;
    bool valid = row < rows;
    T* dst = smem_tile + (size_t)row * D_SMEM + col;
    const T* src = gmem_tile + (size_t)row * D + col;
    unsigned int dst_smem = ptx::cvta_to_shared(dst);
    ptx::cp_async_16B(dst_smem, src, valid);
  }
}

template <int THREADS>
__device__ __forceinline__ void async_load_csq_full_tile(
    float* smem_tile,
    const float* gmem_tile,
    int cols_max) {
  const int tid = threadIdx.x;
  constexpr int floats_per_load = 4;  // 16 bytes
  for (int off = tid * floats_per_load; off < cols_max;
       off += THREADS * floats_per_load) {
    unsigned int dst_smem = ptx::cvta_to_shared(smem_tile + off);
    ptx::cp_async_16B(dst_smem, gmem_tile + off, true);
  }
}

template <int THREADS>
__device__ __forceinline__ void store_csq_tile(
    float* smem_tile,
    const float* gmem_tile,
    int cols,
    int cols_max) {
  const int tid = threadIdx.x;
  for (int i = tid * 4; i < cols_max; i += THREADS * 4) {
    float4 v;
    v.x = (i + 0 < cols) ? gmem_tile[i + 0] : 0.f;
    v.y = (i + 1 < cols) ? gmem_tile[i + 1] : 0.f;
    v.z = (i + 2 < cols) ? gmem_tile[i + 2] : 0.f;
    v.w = (i + 3 < cols) ? gmem_tile[i + 3] : 0.f;
    *reinterpret_cast<float4*>(&smem_tile[i]) = v;
  }
}


template <typename T, int BLOCK_N, int BLOCK_K, int WARPS_PER_CTA, int PIPE_STAGES,
          int N_TILES_PER_CTA = 1, bool ASYNC_CSQ = false, int D_FIXED = 0,
          bool RAW_DIST = false>
__global__ void __launch_bounds__(WARPS_PER_CTA * 32, 1)
assign_sm80_kernel(
    const T* __restrict__ x,            // (B, N, D)
    const T* __restrict__ centroids,    // (B, K, D)
    const float* __restrict__ x_sq,     // (B, N)
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

  // CTA-wide invariants (hoisted above the n_tile loop).
  const int pid_b = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarp;
  const int lane = tid % kWarp;
  const int D_TILE = (D_FIXED > 0) ? D_FIXED : D;
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
  const int k_in_atom_lo = col_in_atom;             // K-half 0 col index in atom
  const int k_in_atom_hi = col_in_atom + 8;         // K-half 1 col index in atom

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
      if constexpr (ASYNC_CSQ) {
        if (k_count == BLOCK_K) {
          async_load_csq_full_tile<THREADS_PER_CTA>(csq_dst, csq_src, BLOCK_K);
        } else {
          store_csq_tile<THREADS_PER_CTA>(csq_dst, csq_src, k_count, BLOCK_K);
        }
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
          const float xs_top = xs_top_cache[m];
          const float xs_bot = xs_bot_cache[m];

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
          const float xs_top = xs_top_cache[m];
          const float xs_bot = xs_bot_cache[m];

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
        const float xs_top = xs_top_cache[m];
        const float xs_bot = xs_bot_cache[m];

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

}  // namespace


// Helper: compute SMEM bytes for a given tile (x_smem + PIPE_STAGES *
// (c_smem + c_sq_smem); x_sq lives in registers).
static inline size_t compute_smem_bytes(int BLOCK_N_, int BLOCK_K_, int D,
                                        int PIPE_STAGES_, size_t elt_sz) {
  int D_SMEM = D + SMEM_PAD;
  return (size_t)BLOCK_N_ * D_SMEM * elt_sz +
         (size_t)PIPE_STAGES_ * BLOCK_K_ * D_SMEM * elt_sz +
         (size_t)PIPE_STAGES_ * BLOCK_K_ * sizeof(float);
}

template <typename T, int BLOCK_N_, int BLOCK_K_, int WARPS_, int STAGES_,
          int N_TILES_ = 1, bool ASYNC_CSQ_ = false, int D_FIXED_ = 0,
          bool RAW_DIST_ = false>
static void launch_typed(
    const at::Tensor& x,
    const at::Tensor& centroids,
    const at::Tensor& x_sq,
    const at::Tensor& c_sq,
    at::Tensor& cluster_ids,
    int B, int N, int K, int D,
    cudaStream_t stream) {
  size_t smem_bytes = compute_smem_bytes(BLOCK_N_, BLOCK_K_, D, STAGES_, sizeof(T));
  auto fn = assign_sm80_kernel<T, BLOCK_N_, BLOCK_K_, WARPS_, STAGES_,
                               N_TILES_, ASYNC_CSQ_, D_FIXED_, RAW_DIST_>;
  if (smem_bytes > 48 * 1024) {
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_bytes));
  }
  // Persistent kernel: each CTA handles N_TILES_ contiguous BLOCK_N_ rows.
  const int rows_per_cta = BLOCK_N_ * N_TILES_;
  dim3 grid((N + rows_per_cta - 1) / rows_per_cta, B);
  dim3 block(WARPS_ * 32);
  fn<<<grid, block, smem_bytes, stream>>>(
      reinterpret_cast<const T*>(x.data_ptr()),
      reinterpret_cast<const T*>(centroids.data_ptr()),
      x_sq.data_ptr<float>(), c_sq.data_ptr<float>(),
      cluster_ids.data_ptr<int32_t>(),
      B, N, K, D);
}

template <typename T, int BLOCK_N_, int BLOCK_K_, int WARPS_, int STAGES_,
          int N_TILES_ = 1, int D_FIXED_ = 0, bool RAW_DIST_ = false>
static void launch_typed_select_csq(
    const at::Tensor& x,
    const at::Tensor& centroids,
    const at::Tensor& x_sq,
    const at::Tensor& c_sq,
    at::Tensor& cluster_ids,
    int B, int N, int K, int D,
    cudaStream_t stream,
    bool async_csq) {
  if (async_csq) {
    launch_typed<T, BLOCK_N_, BLOCK_K_, WARPS_, STAGES_, N_TILES_, true, D_FIXED_, RAW_DIST_>(
        x, centroids, x_sq, c_sq, cluster_ids, B, N, K, D, stream);
  } else {
    launch_typed<T, BLOCK_N_, BLOCK_K_, WARPS_, STAGES_, N_TILES_, false, D_FIXED_, RAW_DIST_>(
        x, centroids, x_sq, c_sq, cluster_ids, B, N, K, D, stream);
  }
}


void launch_assign_sm80(const at::Tensor& x,
                        const at::Tensor& centroids,
                        const at::Tensor& x_sq,
                        const at::Tensor& c_sq,
                        at::Tensor& cluster_ids) {
  TORCH_CHECK(x.is_cuda() && centroids.is_cuda(), "x and centroids must be CUDA tensors");
  TORCH_CHECK(x.dim() == 3 && centroids.dim() == 3, "x and centroids must be 3D (B,N,D)/(B,K,D)");
  TORCH_CHECK(x.scalar_type() == centroids.scalar_type(),
              "x and centroids must share dtype");
  TORCH_CHECK(x_sq.scalar_type() == at::kFloat && c_sq.scalar_type() == at::kFloat,
              "x_sq and c_sq must be fp32");
  TORCH_CHECK(cluster_ids.scalar_type() == at::kInt,
              "cluster_ids must be int32");

  int B = x.size(0);
  int N = x.size(1);
  int D = x.size(2);
  int K = centroids.size(1);
  TORCH_CHECK(centroids.size(0) == B && centroids.size(2) == D,
              "centroids must be (B, K, D) matching x");
  TORCH_CHECK(D % BLOCK_D == 0, "assign_sm80: D must be a multiple of 16 (got ", D, ")");
  TORCH_CHECK(x.is_contiguous() && centroids.is_contiguous() &&
              x_sq.is_contiguous() && c_sq.is_contiguous() &&
              cluster_ids.is_contiguous(),
              "all tensors must be contiguous");

  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Tile selection. "deep" tile (BLOCK_N=64, BLOCK_K=128) wins when K is
  // large because it amortizes the centroid-tile load over fewer N tiles
  // (2× the centroids per K-chunk → half as many K-chunks). The "wide" tile
  // (BLOCK_N=128, BLOCK_K=64) wins on small K / large N because more points
  // are processed per CTA, raising arithmetic intensity per c_tile load.
  size_t elt_sz = x.element_size();

  // Check the device's dynamic-smem-per-block limit. RTX 4090 (sm_89) caps
  // at ~100 KB; H100 at ~228 KB. For very large D we may need the safe
  // kernel.
  int dev = x.device().index();
  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, dev);
  size_t smem_limit = props.sharedMemPerBlockOptin;
  if (smem_limit == 0) smem_limit = props.sharedMemPerBlock;

  // Tile selection. Empirically on Ada (RTX 4090), the wide tile wins on
  // every K we've measured because of its higher mma-to-load instruction
  // ratio. We compile multiple variants and pick the best one that fits
  // in the device's per-block SMEM budget. The "wide-3stage" variant uses
  // 3 cp.async pipeline stages for deeper async overlap; "wide-2stage" is
  // the fallback when 3 stages won't fit. The "deep" variant is compiled
  // in for the rare case where wide doesn't fit and is forced via env.
  static const bool force_deep_env = []() {
    const char* s = std::getenv("FKC_ASSIGN_DEEP_TILE");
    return s && (std::strcmp(s, "1") == 0 || std::strcmp(s, "true") == 0);
  }();

  // SMEM budgets for each variant.
  size_t smem_widek128_2stage = compute_smem_bytes(128, 128, D, 2, elt_sz);
  size_t smem_widek96_2stage = compute_smem_bytes(128, 96, D, 2, elt_sz);
  size_t smem_narrow_4stage = compute_smem_bytes(64, 64, D, 4, elt_sz);
  size_t smem_narrowk32_2stage = compute_smem_bytes(64, 32, D, 2, elt_sz);
  size_t smem_wide_3stage = compute_smem_bytes(128, 64, D, 3, elt_sz);
  size_t smem_wide_2stage = compute_smem_bytes(128, 64, D, 2, elt_sz);
  size_t smem_deep_2stage = compute_smem_bytes(64, 128, D, 2, elt_sz);
  const bool async_csq = (K >= 256);

  // For SVG2-sized large-K work the SMEM budget forces 1 CTA/SM, which
  // gives only 4 warps/SM = 1 warp/scheduler with WARPS=4. Doubling to
  // WARPS=8 gives 2 warps/scheduler for proper latency hiding without
  // reducing total work (each warp does half the M-atoms).
  // BLOCK_N=128, BLOCK_K=128, 8 warps, 2 stages — biggest K-chunk; only
  // fits SMEM when D is small enough.
  auto try_launch_widek128_2_w8 = [&](auto t) -> bool {
    using T = decltype(t);
    if (smem_widek128_2stage > smem_limit) return false;
    launch_typed_select_csq<T, 128, 128, 8, 2>(x, centroids, x_sq, c_sq, cluster_ids,
                                               B, N, K, D, stream, async_csq);
    return true;
  };
  // Persistent variants: each CTA processes N_TILES contiguous BLOCK_N rows.
  // Cuts launch count and lets the K-chunk pipeline + xs/c_sq caches amortize
  // across multiple n-tiles, but the inter-tile cp.async drain costs cycles.
  // Worth it when launch overhead and per-tile prologue/epilogue matter (high
  // grid fan-out, low per-tile work).
  auto try_launch_widek128_2_w8_n2 = [&](auto t) -> bool {
    using T = decltype(t);
    if (smem_widek128_2stage > smem_limit) return false;
    launch_typed_select_csq<T, 128, 128, 8, 2, 2>(x, centroids, x_sq, c_sq, cluster_ids,
                                                  B, N, K, D, stream, async_csq);
    return true;
  };
  // BLOCK_N=128, BLOCK_K=96, 8 warps, 2 stages — bigger K-chunk = longer
  // per-warp mma queue (12 N-atoms), trades pipeline depth for arithmetic
  // throughput.
  auto try_launch_widek96_2_w8 = [&](auto t) -> bool {
    using T = decltype(t);
    if (smem_widek96_2stage > smem_limit) return false;
    launch_typed_select_csq<T, 128, 96, 8, 2>(x, centroids, x_sq, c_sq, cluster_ids,
                                              B, N, K, D, stream, async_csq);
    return true;
  };
  auto try_launch_widek96_2_w8_n2 = [&](auto t) -> bool {
    using T = decltype(t);
    if (smem_widek96_2stage > smem_limit) return false;
    launch_typed_select_csq<T, 128, 96, 8, 2, 2>(x, centroids, x_sq, c_sq, cluster_ids,
                                                 B, N, K, D, stream, async_csq);
    return true;
  };
  auto try_launch_widek96_2_w8_n2_d128 = [&](auto t) -> bool {
    using T = decltype(t);
    if (D != 128 || K < 2048 || smem_widek96_2stage > smem_limit) return false;
    // Raw distance trims the hot epilogue and lets the Python loop skip x_sq.
    // K=2048 benchmarked slower with it, so keep that shape on clamped dist.
    if (K >= 4096) {
      launch_typed_select_csq<T, 128, 96, 8, 2, 2, 128, true>(
          x, centroids, x_sq, c_sq, cluster_ids, B, N, K, D, stream, async_csq);
    } else {
      launch_typed_select_csq<T, 128, 96, 8, 2, 2, 128, false>(
          x, centroids, x_sq, c_sq, cluster_ids, B, N, K, D, stream, async_csq);
    }
    return true;
  };
  auto try_launch_widek128_2_w8_n4 = [&](auto t) -> bool {
    using T = decltype(t);
    if (smem_widek128_2stage > smem_limit) return false;
    launch_typed_select_csq<T, 128, 128, 8, 2, 4>(x, centroids, x_sq, c_sq, cluster_ids,
                                                  B, N, K, D, stream, async_csq);
    return true;
  };
  auto try_launch_widek96_2_w8_n4 = [&](auto t) -> bool {
    using T = decltype(t);
    if (smem_widek96_2stage > smem_limit) return false;
    launch_typed_select_csq<T, 128, 96, 8, 2, 4>(x, centroids, x_sq, c_sq, cluster_ids,
                                                 B, N, K, D, stream, async_csq);
    return true;
  };
  // BLOCK_N=64, BLOCK_K=64, 4 warps, 4 stages — deepest async pipeline that
  // fits within Ada's 100 KB SMEM. Useful for very-large K where pipeline
  // depth dominates over per-CTA arithmetic intensity.
  auto try_launch_narrow_4 = [&](auto t) -> bool {
    using T = decltype(t);
    if (smem_narrow_4stage > smem_limit) return false;
    launch_typed_select_csq<T, 64, 64, 4, 4>(x, centroids, x_sq, c_sq, cluster_ids,
                                             B, N, K, D, stream, async_csq);
    return true;
  };
  // BLOCK_N=64, BLOCK_K=32, 4 warps, 2 stages — small enough (~34 KB) for
  // 2 CTAs/SM on Ada (100 KB / SM). Total 8 warps/SM = 2 warps/scheduler,
  // same as 8w 1-CTA but with INDEPENDENT CTAs (no shared __syncthreads),
  // closer to Triton's BN=128 BK=32 num_warps=4 num_stages=1 autotune for
  // K >= 2K. Set FKC_NARROW=1 to force this path.
  auto try_launch_narrowk32_2_w4 = [&](auto t) -> bool {
    using T = decltype(t);
    if (smem_narrowk32_2stage > smem_limit) return false;
    launch_typed_select_csq<T, 64, 32, 4, 2>(x, centroids, x_sq, c_sq, cluster_ids,
                                             B, N, K, D, stream, async_csq);
    return true;
  };
  auto try_launch_narrowk32_2_w4_n2 = [&](auto t) -> bool {
    using T = decltype(t);
    if (smem_narrowk32_2stage > smem_limit) return false;
    launch_typed_select_csq<T, 64, 32, 4, 2, 2>(x, centroids, x_sq, c_sq, cluster_ids,
                                                B, N, K, D, stream, async_csq);
    return true;
  };
  auto try_launch_wide_3_w8 = [&](auto t) -> bool {
    using T = decltype(t);
    if (smem_wide_3stage > smem_limit) return false;
    launch_typed_select_csq<T, 128, 64, 8, 3>(x, centroids, x_sq, c_sq, cluster_ids,
                                              B, N, K, D, stream, async_csq);
    return true;
  };
  auto try_launch_wide_3_w8_n2 = [&](auto t) -> bool {
    using T = decltype(t);
    if (smem_wide_3stage > smem_limit) return false;
    launch_typed_select_csq<T, 128, 64, 8, 3, 2>(x, centroids, x_sq, c_sq, cluster_ids,
                                                 B, N, K, D, stream, async_csq);
    return true;
  };
  auto try_launch_wide_3_w4 = [&](auto t) -> bool {
    using T = decltype(t);
    if (smem_wide_3stage > smem_limit) return false;
    launch_typed_select_csq<T, 128, 64, 4, 3>(x, centroids, x_sq, c_sq, cluster_ids,
                                              B, N, K, D, stream, async_csq);
    return true;
  };
  auto try_launch_wide_2_w4 = [&](auto t) -> bool {
    using T = decltype(t);
    if (smem_wide_2stage > smem_limit) return false;
    launch_typed_select_csq<T, 128, 64, 4, 2>(x, centroids, x_sq, c_sq, cluster_ids,
                                              B, N, K, D, stream, async_csq);
    return true;
  };
  auto try_launch_deep_2_w4 = [&](auto t) -> bool {
    using T = decltype(t);
    if (smem_deep_2stage > smem_limit) return false;
    launch_typed_select_csq<T, 64, 128, 4, 2>(x, centroids, x_sq, c_sq, cluster_ids,
                                              B, N, K, D, stream, async_csq);
    return true;
  };

  // Tile selection by K bucket. Empirically on Ada (RTX 4090):
  //   K < 512  : wide-w4-3stage (more M-atoms per warp = higher arithmetic
  //              intensity, fewer warps means each warp gets long mma runs).
  //   K >= 512 : wide-w8-3stage (1 CTA/SM is forced by SMEM; doubling
  //              warps gives 2 warps/scheduler for latency hiding).
  // The narrow_4 (BLOCK_N=64 BLOCK_K=64, 4 stages) tile is kept as a
  // last-resort fallback before deep when SMEM is tight on unusual shapes.
  bool prefer_w8 = (K >= 128);

  // Persistent N-tile knob. Default 2 — empirically wins or ties N_TILES=1
  // across med/big/huge/mega shapes on Ada (RTX 4090). Halving the launch
  // count amortizes per-tile prologue (x_tile load + xs/c_sq cache + epilogue)
  // over 2 N-tiles per CTA; the inter-tile cp.async drain costs less than
  // the saved setup. Override with FKC_NTILES={1,2,4} to re-A/B.
  static const int n_tiles_env = []() {
    const char* s = std::getenv("FKC_NTILES");
    if (!s) return 2;
    int v = std::atoi(s);
    if (v == 1) return 1;
    if (v == 4) return 4;
    return 2;
  }();

  // 3-stage wide tile experiment knob. FKC_WIDE3=1 forces BN=128 BK=64 8w
  // 3-stage instead of the BK=128/96 2-stage default. Smaller BK fits a
  // deeper async pipeline; on long-K shapes this may overlap loads better.
  static const bool force_wide3_env = []() {
    const char* s = std::getenv("FKC_WIDE3");
    return s && (std::strcmp(s, "1") == 0 || std::strcmp(s, "true") == 0);
  }();

  // 4-warp wide tile experiment knob. FKC_W4=1 forces BN=128 BK=64 4w
  // 2-stage instead of the 8w default. Halves warps/CTA → 2 atoms per warp
  // (M_ATOMS=2) and longer mma queue per warp; closer to Triton's
  // num_warps=4 autotune choice for K>=2K.
  static const bool force_w4_env = []() {
    const char* s = std::getenv("FKC_W4");
    return s && (std::strcmp(s, "1") == 0 || std::strcmp(s, "true") == 0);
  }();

  // 2-CTAs/SM narrow tile knob. FKC_NARROW=1 forces BN=64 BK=32 4w 2-stage
  // (~34 KB SMEM/CTA fits 2 CTAs/SM on Ada, vs current 87 KB → 1 CTA/SM).
  // Effective warps/SM identical (8) but split across 2 independent CTAs
  // — cuts the inter-K-chunk barrier serialization that 8w 1-CTA suffers.
  static const bool force_narrow_env = []() {
    const char* s = std::getenv("FKC_NARROW");
    return s && (std::strcmp(s, "1") == 0 || std::strcmp(s, "true") == 0);
  }();

  bool launched = false;
  if (force_deep_env) {
    if (x.scalar_type() == at::kHalf)             launched = try_launch_deep_2_w4(__half{});
    else if (x.scalar_type() == at::kBFloat16)    launched = try_launch_deep_2_w4(__nv_bfloat16{});
  } else {
    auto run = [&](auto t) {
      using T = decltype(t);
      if (prefer_w8) {
        if (force_narrow_env) {
          if (n_tiles_env >= 2) {
            if (try_launch_narrowk32_2_w4_n2(t)) return true;
          }
          if (try_launch_narrowk32_2_w4(t)) return true;
        }
        if (force_w4_env) {
          if (try_launch_wide_3_w4(t))      return true;
          if (try_launch_wide_2_w4(t))      return true;
        }
        if (force_wide3_env) {
          if (n_tiles_env >= 2) {
            if (try_launch_wide_3_w8_n2(t)) return true;
          }
          if (try_launch_wide_3_w8(t))      return true;
        }
        // Default (n_tiles_env=2) tries the N_TILES=2 wide variants first;
        // falls through to N_TILES=1 path on SMEM miss or when env forces 1.
        // BK=112 (intermediate, fits SMEM where 128 doesn't) was tested
        // and regressed 16% on mega vs BK=96 — register pressure from 14
        // N-atoms × 2 acc regs + 14 B regs / thread tipped past nvcc's
        // sweet spot. Reverted; left history in commit.
        if (n_tiles_env == 2) {
          if (try_launch_widek128_2_w8_n2(t)) return true;
          if (try_launch_widek96_2_w8_n2_d128(t)) return true;
          if (try_launch_widek96_2_w8_n2(t))  return true;
        } else if (n_tiles_env == 4) {
          if (try_launch_widek128_2_w8_n4(t)) return true;
          if (try_launch_widek96_2_w8_n4(t))  return true;
        }
        return try_launch_widek128_2_w8(t) ||
               try_launch_widek96_2_w8(t) ||
               try_launch_wide_3_w8(t) ||
               try_launch_wide_3_w4(t) ||
               try_launch_wide_2_w4(t) ||
               try_launch_narrow_4(t) ||
               try_launch_deep_2_w4(t);
      } else {
        return try_launch_wide_3_w4(t) ||
               try_launch_wide_2_w4(t) ||
               try_launch_narrow_4(t) ||
               try_launch_deep_2_w4(t);
      }
    };
    if (x.scalar_type() == at::kHalf)            launched = run(__half{});
    else if (x.scalar_type() == at::kBFloat16)   launched = run(__nv_bfloat16{});
    else TORCH_CHECK(false, "assign_sm80 requires fp16 or bf16 input");
  }

  if (!launched) {
    // No tile fits; fall back to the safe kernel.
    launch_assign_safe(x, centroids, x_sq, c_sq, cluster_ids);
    return;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


}  // namespace assign
}  // namespace fkc
