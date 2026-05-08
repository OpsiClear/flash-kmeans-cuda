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
// - D must be 3..15 or a multiple of 16. D=3..15 are zero-padded to a D=16
//   SMEM tile.
// - Bank conflicts: scalar loads from a (BLOCK_N, D) row-major SMEM tile have
//   conflicts when D is a power of 2. We accept this for now (correctness
//   first); a future patch can pad the row stride.

#include "assign.h"
#include "assign_kernel_launch.h"
// assign_sm80_kernel template definition. Moved to a .cuh header so that
// assign_policy.cu (a separate TU) can instantiate its function pointers.
#include "assign_sm80_kernel.cuh"
#include "assign_policy.h"
#include "assign_autotune.h"
#include "../common/torch_cuda_includes.h"

namespace fkc {
namespace assign {

void launch_assign_sm80(const at::Tensor& x,
                        const at::Tensor& centroids,
                        const at::Tensor& x_sq,
                        const at::Tensor& c_sq,
                        at::Tensor& cluster_ids) {
  TORCH_CHECK(x.is_cuda() && centroids.is_cuda(),
              "x and centroids must be CUDA tensors");
  TORCH_CHECK(x.dim() == 3 && centroids.dim() == 3,
              "x and centroids must be 3D (B,N,D)/(B,K,D)");
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
  TORCH_CHECK((D >= 3 && D < BLOCK_D) || D % BLOCK_D == 0,
              "assign_sm80: D must be 3..15 or a multiple of 16 (got ", D, ")");
  TORCH_CHECK(x.is_contiguous() && centroids.is_contiguous() &&
              x_sq.is_contiguous() && c_sq.is_contiguous() &&
              cluster_ids.is_contiguous(),
              "all tensors must be contiguous");

  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int dev = x.device().index();
  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, dev);
  size_t smem_limit = props.sharedMemPerBlockOptin;
  if (smem_limit == 0) smem_limit = props.sharedMemPerBlock;

  LaunchCtx ctx{
    x, centroids, x_sq, c_sq, cluster_ids,
    B, N, K, D,
    static_cast<size_t>(x.element_size()), smem_limit,
    /*async_csq=*/(K >= 256),
    stream,
  };

  EnvKnobs knobs = read_env_knobs();

  int dtype_idx = dtype_index_of(x.scalar_type());
  TORCH_CHECK(dtype_idx >= 0, "assign_sm80 requires fp16 or bf16 input");

  VariantView candidates{nullptr, 0};
  if (knobs.has_force_override()) {
    candidates = build_forced_candidates(knobs, ctx);
  } else {
    AutotuneKey key{ dtype_idx, d_index_of(D), k_bucket_of(K) };
    candidates = autotune_cache().get_or_probe(
        key, ctx,
        static_policy(dtype_idx, key.d_idx, key.k_bucket, knobs.n_tiles_override),
        knobs.autotune, knobs.verbose);
  }

  bool launched = false;
  bool is_fp16 = (x.scalar_type() == at::kHalf);
  for (size_t i = 0; i < candidates.size(); ++i) {
    const Variant* v = candidates[i];
    if (!v) break;
    bool ok = is_fp16 ? v->try_fp16(ctx) : v->try_bf16(ctx);
    if (ok) { launched = true; break; }
  }

  if (!launched) {
    launch_assign_safe(x, centroids, x_sq, c_sq, cluster_ids);
    return;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


}  // namespace assign
}  // namespace fkc
