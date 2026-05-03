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
#include "assign_kernel_launch.h"
// assign_sm80_kernel template definition. Moved to a .cuh header so that
// assign_policy.cu (a separate TU) can instantiate its function pointers.
#include "assign_sm80_kernel.cuh"
#include "../common/torch_cuda_includes.h"

#include <cstdlib>
#include <cstring>

namespace fkc {
namespace assign {

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
  auto try_launch_widek128_2_w8_d128 = [&](auto t) -> bool {
    using T = decltype(t);
    if (D != 128 || smem_widek128_2stage > smem_limit) return false;
    launch_typed_select_csq<T, 128, 128, 8, 2, 1, 128, true>(
        x, centroids, x_sq, c_sq, cluster_ids, B, N, K, D, stream, async_csq);
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
  auto try_launch_widek128_2_w8_n2_d64 = [&](auto t) -> bool {
    using T = decltype(t);
    if (D != 64 || smem_widek128_2stage > smem_limit) return false;
    launch_typed_select_csq<T, 128, 128, 8, 2, 2, 64, true>(
        x, centroids, x_sq, c_sq, cluster_ids, B, N, K, D, stream, async_csq);
    return true;
  };
  auto try_launch_widek128_2_w8_n2_d96 = [&](auto t) -> bool {
    using T = decltype(t);
    if (D != 96 || smem_widek128_2stage > smem_limit) return false;
    launch_typed_select_csq<T, 128, 128, 8, 2, 2, 96, true>(
        x, centroids, x_sq, c_sq, cluster_ids, B, N, K, D, stream, async_csq);
    return true;
  };
  auto try_launch_widek128_2_w8_n2_d128 = [&](auto t) -> bool {
    using T = decltype(t);
    if (D != 128 || smem_widek128_2stage > smem_limit) return false;
    launch_typed_select_csq<T, 128, 128, 8, 2, 2, 128, true>(
        x, centroids, x_sq, c_sq, cluster_ids, B, N, K, D, stream, async_csq);
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
  auto try_launch_widek96_2_w8_d128 = [&](auto t) -> bool {
    using T = decltype(t);
    if (D != 128 || K < 256 || smem_widek96_2stage > smem_limit) return false;
    launch_typed_select_csq<T, 128, 96, 8, 2, 1, 128, true>(
        x, centroids, x_sq, c_sq, cluster_ids, B, N, K, D, stream, async_csq);
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
    if (D != 128 || K < 256 || smem_widek96_2stage > smem_limit) return false;
    // Raw distance trims the hot epilogue and lets the Python loop skip x_sq.
    launch_typed_select_csq<T, 128, 96, 8, 2, 2, 128, true>(
        x, centroids, x_sq, c_sq, cluster_ids, B, N, K, D, stream, async_csq);
    return true;
  };
  auto try_launch_widek128_2_w8_n4 = [&](auto t) -> bool {
    using T = decltype(t);
    if (smem_widek128_2stage > smem_limit) return false;
    launch_typed_select_csq<T, 128, 128, 8, 2, 4>(x, centroids, x_sq, c_sq, cluster_ids,
                                                  B, N, K, D, stream, async_csq);
    return true;
  };
  auto try_launch_widek128_2_w8_n4_d128 = [&](auto t) -> bool {
    using T = decltype(t);
    if (D != 128 || smem_widek128_2stage > smem_limit) return false;
    launch_typed_select_csq<T, 128, 128, 8, 2, 4, 128, true>(
        x, centroids, x_sq, c_sq, cluster_ids, B, N, K, D, stream, async_csq);
    return true;
  };
  auto try_launch_widek96_2_w8_n4 = [&](auto t) -> bool {
    using T = decltype(t);
    if (smem_widek96_2stage > smem_limit) return false;
    launch_typed_select_csq<T, 128, 96, 8, 2, 4>(x, centroids, x_sq, c_sq, cluster_ids,
                                                 B, N, K, D, stream, async_csq);
    return true;
  };
  auto try_launch_widek96_2_w8_n4_d128 = [&](auto t) -> bool {
    using T = decltype(t);
    if (D != 128 || K < 256 || smem_widek96_2stage > smem_limit) return false;
    launch_typed_select_csq<T, 128, 96, 8, 2, 4, 128, true>(
        x, centroids, x_sq, c_sq, cluster_ids, B, N, K, D, stream, async_csq);
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
  auto try_launch_narrowk32_2_w4_n2_d192 = [&](auto t) -> bool {
    using T = decltype(t);
    if (D != 192 || smem_narrowk32_2stage > smem_limit) return false;
    launch_typed_select_csq<T, 64, 32, 4, 2, 2, 192, true>(
        x, centroids, x_sq, c_sq, cluster_ids, B, N, K, D, stream, async_csq);
    return true;
  };
  auto try_launch_narrowk32_2_w4_n2_d224 = [&](auto t) -> bool {
    using T = decltype(t);
    if (D != 224 || smem_narrowk32_2stage > smem_limit) return false;
    launch_typed_select_csq<T, 64, 32, 4, 2, 2, 224, true>(
        x, centroids, x_sq, c_sq, cluster_ids, B, N, K, D, stream, async_csq);
    return true;
  };
  auto try_launch_narrowk32_2_w4_n2_d256 = [&](auto t) -> bool {
    using T = decltype(t);
    if (D != 256 || smem_narrowk32_2stage > smem_limit) return false;
    launch_typed_select_csq<T, 64, 32, 4, 2, 2, 256, true>(
        x, centroids, x_sq, c_sq, cluster_ids, B, N, K, D, stream, async_csq);
    return true;
  };
  auto try_launch_narrowk32_2_w4_n2_d320 = [&](auto t) -> bool {
    using T = decltype(t);
    if (D != 320 || smem_narrowk32_2stage > smem_limit) return false;
    launch_typed_select_csq<T, 64, 32, 4, 2, 2, 320, true>(
        x, centroids, x_sq, c_sq, cluster_ids, B, N, K, D, stream, async_csq);
    return true;
  };
  auto try_launch_narrowk32_2_w4_n2_d384 = [&](auto t) -> bool {
    using T = decltype(t);
    if (D != 384 || smem_narrowk32_2stage > smem_limit) return false;
    launch_typed_select_csq<T, 64, 32, 4, 2, 2, 384, true>(
        x, centroids, x_sq, c_sq, cluster_ids, B, N, K, D, stream, async_csq);
    return true;
  };
  auto try_launch_narrowk32_2_w4_n4 = [&](auto t) -> bool {
    using T = decltype(t);
    if (smem_narrowk32_2stage > smem_limit) return false;
    launch_typed_select_csq<T, 64, 32, 4, 2, 4>(x, centroids, x_sq, c_sq, cluster_ids,
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

  // Persistent N-tile knob. This is now D-aware: D=128 defaults to
  // N_TILES=1, while other tested D values keep N_TILES=2 so narrow
  // large-D fallbacks remain available.
  // Override with FKC_NTILES={1,2,4} to re-A/B.
  static const int n_tiles_env = []() {
    const char* s = std::getenv("FKC_NTILES");
    if (!s) return 0;
    int v = std::atoi(s);
    if (v == 1) return 1;
    if (v == 2) return 2;
    if (v == 4) return 4;
    return 0;
  }();
  auto default_n_tiles_for_d = [&](int d) -> int {
    if (d == 128) return 1;
    return 2;
  };
  const int n_tiles_choice = (n_tiles_env != 0)
      ? n_tiles_env
      : default_n_tiles_for_d(D);

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
          if (n_tiles_env == 4) {
            if (try_launch_narrowk32_2_w4_n4(t)) return true;
          } else if (n_tiles_env >= 2) {
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
        // Default auto usually tries the N_TILES=2 wide variants first;
        // falls through to N_TILES=1 path on SMEM miss or when env forces 1.
        // BK=112 (intermediate, fits SMEM where 128 doesn't) was tested
        // and regressed 16% on mega vs BK=96 — register pressure from 14
        // N-atoms × 2 acc regs + 14 B regs / thread tipped past nvcc's
        // sweet spot. Reverted; left history in commit.
        if (n_tiles_choice == 2) {
          switch (D) {
            case 64:
              if (try_launch_widek128_2_w8_n2_d64(t)) return true;
              if (try_launch_widek128_2_w8_n2(t)) return true;
              if (try_launch_widek96_2_w8_n2(t)) return true;
              break;
            case 96:
              if (try_launch_widek128_2_w8_n2_d96(t)) return true;
              if (try_launch_widek128_2_w8_n2(t)) return true;
              if (try_launch_widek96_2_w8_n2(t)) return true;
              break;
            case 128:
              if (try_launch_widek128_2_w8_n2_d128(t)) return true;
              if (try_launch_widek128_2_w8_n2(t)) return true;
              if (try_launch_widek96_2_w8_n2_d128(t)) return true;
              if (try_launch_widek96_2_w8_n2(t)) return true;
              break;
            default:
              if (try_launch_widek128_2_w8_n2(t)) return true;
              if (try_launch_widek96_2_w8_n2(t)) return true;
              break;
          }
        } else if (n_tiles_choice == 4) {
          if (try_launch_widek128_2_w8_n4(t)) return true;
          if (D == 128 && try_launch_widek128_2_w8_n4_d128(t)) return true;
          if (try_launch_widek96_2_w8_n4_d128(t)) return true;
          if (try_launch_widek96_2_w8_n4(t))  return true;
        }
        if (D == 128 && try_launch_widek128_2_w8_d128(t)) return true;
        return try_launch_widek128_2_w8(t) ||
               try_launch_widek96_2_w8_d128(t) ||
               try_launch_widek96_2_w8(t) ||
               try_launch_wide_3_w8(t) ||
               try_launch_wide_3_w4(t) ||
               try_launch_wide_2_w4(t) ||
               (n_tiles_choice == 4 && try_launch_narrowk32_2_w4_n4(t)) ||
               (n_tiles_choice == 2 && try_launch_narrowk32_2_w4_n2_d192(t)) ||
               (n_tiles_choice == 2 && try_launch_narrowk32_2_w4_n2_d224(t)) ||
               (n_tiles_choice == 2 && try_launch_narrowk32_2_w4_n2_d256(t)) ||
               (n_tiles_choice == 2 && try_launch_narrowk32_2_w4_n2_d320(t)) ||
               (n_tiles_choice == 2 && try_launch_narrowk32_2_w4_n2_d384(t)) ||
               (n_tiles_choice == 2 && try_launch_narrowk32_2_w4_n2(t)) ||
               try_launch_narrowk32_2_w4(t) ||
               try_launch_narrow_4(t) ||
               try_launch_deep_2_w4(t);
      } else {
        return try_launch_wide_3_w4(t) ||
               try_launch_wide_2_w4(t) ||
               (n_tiles_choice == 4 && try_launch_narrowk32_2_w4_n4(t)) ||
               (n_tiles_choice == 2 && try_launch_narrowk32_2_w4_n2_d192(t)) ||
               (n_tiles_choice == 2 && try_launch_narrowk32_2_w4_n2_d224(t)) ||
               (n_tiles_choice == 2 && try_launch_narrowk32_2_w4_n2_d256(t)) ||
               (n_tiles_choice == 2 && try_launch_narrowk32_2_w4_n2_d320(t)) ||
               (n_tiles_choice == 2 && try_launch_narrowk32_2_w4_n2_d384(t)) ||
               (n_tiles_choice == 2 && try_launch_narrowk32_2_w4_n2(t)) ||
               try_launch_narrowk32_2_w4(t) ||
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
