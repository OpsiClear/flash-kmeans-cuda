// Tensor-core-free Euclidean assignment kernel.
//
// One thread per (batch, point); each thread streams the K dimension and
// computes ||x - c||^2 in fp32 against its single point. Inputs may be
// fp16/bf16/fp32; operands are cast to fp32 in the inner loop. This is the
// correctness baseline (Phase A) — Phase B's mma/wgmma kernels are validated
// against it before being made the default.
//
// Despite "no tensor cores," this is competitive on small/medium K because:
//   - Each centroid is read from gmem once per CTA-resident point group
//     (relies on L1/L2 sharing across threads in the same warp).
//   - The dot-product loop is fully unrolled by D for D % 4 == 0.
//   - A simple per-warp reduction collapses the 32 candidate K-values per
//     iteration into per-thread bests.

#include "assign.h"
#include "assign_common.cuh"
#include "../common/arch.cuh"

#include "../common/torch_cuda_includes.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <cstdint>

namespace fkc {
namespace assign {

namespace {

constexpr int BLOCK_N = 64;
constexpr int THREADS_PER_CTA = 64;     // one thread per point row in BLOCK_N

__device__ __forceinline__ float to_fp32(__half h)         { return __half2float(h); }
__device__ __forceinline__ float to_fp32(__nv_bfloat16 b)  { return __bfloat162float(b); }
__device__ __forceinline__ float to_fp32(float f)          { return f; }

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_CTA, 4)
assign_safe_kernel(
    const T* __restrict__ x,            // (B, N, D)
    const T* __restrict__ centroids,    // (B, K, D)
    const float* __restrict__ x_sq,     // (B, N)
    const float* __restrict__ c_sq,     // (B, K)
    int32_t* __restrict__ cluster_ids,  // (B, N)
    int B, int N, int K, int D) {
  const int pid_n = blockIdx.x;
  const int pid_b = blockIdx.y;
  const int tid = threadIdx.x;

  const int n_idx = pid_n * BLOCK_N + tid;
  if (n_idx >= N) return;

  const T* x_row = x + (size_t)pid_b * N * D + (size_t)n_idx * D;
  const float xs = x_sq[(size_t)pid_b * N + n_idx];

  Best me{FLT_MAX, 0};

  // Stream over K. For each centroid, dot-product against this point.
  for (int k = 0; k < K; ++k) {
    const T* c_row = centroids + (size_t)pid_b * K * D + (size_t)k * D;
    float dot = 0.f;
    int d = 0;
    // Unrolled inner loop, fp32 accumulation.
    #pragma unroll 4
    for (; d + 4 <= D; d += 4) {
      dot += to_fp32(x_row[d + 0]) * to_fp32(c_row[d + 0]);
      dot += to_fp32(x_row[d + 1]) * to_fp32(c_row[d + 1]);
      dot += to_fp32(x_row[d + 2]) * to_fp32(c_row[d + 2]);
      dot += to_fp32(x_row[d + 3]) * to_fp32(c_row[d + 3]);
    }
    for (; d < D; ++d) {
      dot += to_fp32(x_row[d]) * to_fp32(c_row[d]);
    }
    float cs = c_sq[(size_t)pid_b * K + k];
    float dist = to_dist(dot, xs, cs);
    update_best(me, dist, k);
  }

  cluster_ids[(size_t)pid_b * N + n_idx] = me.idx;
}

template <typename T>
void launch_assign_safe_typed(
    const at::Tensor& x,
    const at::Tensor& centroids,
    const at::Tensor& x_sq,
    const at::Tensor& c_sq,
    at::Tensor& cluster_ids) {
  int B = x.size(0);
  int N = x.size(1);
  int D = x.size(2);
  int K = centroids.size(1);

  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid((N + BLOCK_N - 1) / BLOCK_N, B);
  dim3 block(THREADS_PER_CTA);
  assign_safe_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const T*>(x.data_ptr()),
      reinterpret_cast<const T*>(centroids.data_ptr()),
      x_sq.data_ptr<float>(),
      c_sq.data_ptr<float>(),
      cluster_ids.data_ptr<int32_t>(),
      B, N, K, D);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace


// Public dispatch: routes any dtype to the safe kernel.
void launch_assign_safe(const at::Tensor& x,
                        const at::Tensor& centroids,
                        const at::Tensor& x_sq,
                        const at::Tensor& c_sq,
                        at::Tensor& cluster_ids) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(x.is_contiguous() && centroids.is_contiguous() &&
              x_sq.is_contiguous() && c_sq.is_contiguous() &&
              cluster_ids.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(cluster_ids.scalar_type() == at::kInt, "cluster_ids must be int32");

  if (x.scalar_type() == at::kHalf) {
    launch_assign_safe_typed<__half>(x, centroids, x_sq, c_sq, cluster_ids);
  } else if (x.scalar_type() == at::kBFloat16) {
    launch_assign_safe_typed<__nv_bfloat16>(x, centroids, x_sq, c_sq, cluster_ids);
  } else if (x.scalar_type() == at::kFloat) {
    launch_assign_safe_typed<float>(x, centroids, x_sq, c_sq, cluster_ids);
  } else {
    TORCH_CHECK(false, "assign_safe: unsupported dtype ", x.scalar_type());
  }
}

}  // namespace assign
}  // namespace fkc
