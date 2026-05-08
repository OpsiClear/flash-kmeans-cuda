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
#include <ATen/ops/sort.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <cstdint>

namespace fkc {
namespace assign {

namespace {

constexpr int BLOCK_N = 64;
constexpr int THREADS_PER_CTA = 64;     // one thread per point row in BLOCK_N
constexpr int SMALL_D_BLOCK_N = 128;
constexpr int SMALL_D_BLOCK_K = 512;
constexpr int D1_SORT_THREADS = 256;

__device__ __forceinline__ float to_fp32(__half h)         { return __half2float(h); }
__device__ __forceinline__ float to_fp32(__nv_bfloat16 b)  { return __bfloat162float(b); }
__device__ __forceinline__ float to_fp32(float f)          { return f; }

__device__ __forceinline__ void update_best_1d(float& best_dist, int& best_idx,
                                               float dist, int idx) {
  if (dist < best_dist || (dist == best_dist && idx < best_idx)) {
    best_dist = dist;
    best_idx = idx;
  }
}

__device__ __forceinline__ void consider_d1_sorted_group(
    const float* __restrict__ vals,
    const int64_t* __restrict__ idxs,
    int K,
    int pos,
    float xv,
    float& best_dist,
    int& best_idx) {
  const float cv = vals[pos];
  int first = pos;
  while (first > 0 && vals[first - 1] == cv) {
    --first;
  }
  int last = pos;
  while ((last + 1) < K && vals[last + 1] == cv) {
    ++last;
  }

  int min_idx = static_cast<int>(idxs[first]);
  for (int p = first + 1; p <= last; ++p) {
    const int idx = static_cast<int>(idxs[p]);
    if (idx < min_idx) min_idx = idx;
  }
  update_best_1d(best_dist, best_idx, cv * cv - 2.0f * xv * cv, min_idx);
}

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

template <typename T, int D_FIXED>
__global__ void __launch_bounds__(SMALL_D_BLOCK_N, 4)
assign_small_d_kernel(
    const T* __restrict__ x,            // (B, N, D_FIXED)
    const T* __restrict__ centroids,    // (B, K, D_FIXED)
    const float* __restrict__ c_sq,     // (B, K)
    int32_t* __restrict__ cluster_ids,  // (B, N)
    int N, int K) {
  static_assert(D_FIXED > 0 && D_FIXED < 16, "small-D kernel handles 1 <= D < 16");
  const int pid_b = blockIdx.y;
  const int tid = threadIdx.x;
  const int n_idx = blockIdx.x * SMALL_D_BLOCK_N + tid;
  const bool active = n_idx < N;

  const T* x_row = x + (size_t)pid_b * N * D_FIXED + (size_t)n_idx * D_FIXED;
  float x_reg[D_FIXED];
  #pragma unroll
  for (int d = 0; d < D_FIXED; ++d) {
    x_reg[d] = active ? to_fp32(x_row[d]) : 0.0f;
  }

  Best me{FLT_MAX, 0};
  const T* c_base = centroids + (size_t)pid_b * K * D_FIXED;
  const float* cs_base = c_sq + (size_t)pid_b * K;

  for (int k = 0; k < K; ++k) {
    const T* c_row = c_base + (size_t)k * D_FIXED;
    float dot = 0.f;
    #pragma unroll
    for (int d = 0; d < D_FIXED; ++d) {
      dot += x_reg[d] * to_fp32(c_row[d]);
    }
    update_best(me, cs_base[k] - 2.0f * dot, k);
  }

  if (active) {
    cluster_ids[(size_t)pid_b * N + n_idx] = me.idx;
  }
}

template <typename T, int D_FIXED, int TILE_N, int BLOCK_K>
__global__ void __launch_bounds__(TILE_N, 4)
assign_small_d_tiled_kernel(
    const T* __restrict__ x,            // (B, N, D_FIXED)
    const T* __restrict__ centroids,    // (B, K, D_FIXED)
    const float* __restrict__ c_sq,     // (B, K)
    int32_t* __restrict__ cluster_ids,  // (B, N)
    int N, int K) {
  static_assert(D_FIXED > 0 && D_FIXED <= 7, "tiled small-D kernel handles 1 <= D <= 7");
  const int pid_b = blockIdx.y;
  const int tid = threadIdx.x;
  const int n_idx = blockIdx.x * TILE_N + tid;
  const bool active = n_idx < N;

  extern __shared__ unsigned char smem_raw[];
  T* c_tile = reinterpret_cast<T*>(smem_raw);
  float* cs_tile = reinterpret_cast<float*>(c_tile + (size_t)BLOCK_K * D_FIXED);

  const int safe_n_idx = active ? n_idx : 0;
  const T* x_row = x + (size_t)pid_b * N * D_FIXED + (size_t)safe_n_idx * D_FIXED;
  float x_reg[D_FIXED];
  #pragma unroll
  for (int d = 0; d < D_FIXED; ++d) {
    x_reg[d] = active ? to_fp32(x_row[d]) : 0.0f;
  }

  Best me{FLT_MAX, 0};
  const T* c_base = centroids + (size_t)pid_b * K * D_FIXED;
  const float* cs_base = c_sq + (size_t)pid_b * K;

  for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
    const int kt = min(BLOCK_K, K - k0);
    const int c_elts = kt * D_FIXED;

    for (int off = tid; off < c_elts; off += TILE_N) {
      c_tile[off] = c_base[(size_t)k0 * D_FIXED + off];
    }
    for (int off = tid; off < kt; off += TILE_N) {
      cs_tile[off] = cs_base[k0 + off];
    }
    __syncthreads();

    if (active) {
      #pragma unroll 4
      for (int kk = 0; kk < kt; ++kk) {
        const T* c_row = c_tile + kk * D_FIXED;
        float dot = 0.f;
        #pragma unroll
        for (int d = 0; d < D_FIXED; ++d) {
          dot += x_reg[d] * to_fp32(c_row[d]);
        }
        update_best(me, cs_tile[kk] - 2.0f * dot, k0 + kk);
      }
    }
    __syncthreads();
  }

  if (active) {
    cluster_ids[(size_t)pid_b * N + n_idx] = me.idx;
  }
}

template <typename T, int D_FIXED>
void launch_assign_small_d_fixed(
    const at::Tensor& x,
    const at::Tensor& centroids,
    const at::Tensor& c_sq,
    at::Tensor& cluster_ids) {
  int B = x.size(0);
  int N = x.size(1);
  int K = centroids.size(1);

  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid((N + SMALL_D_BLOCK_N - 1) / SMALL_D_BLOCK_N, B);
  dim3 block(SMALL_D_BLOCK_N);
  assign_small_d_kernel<T, D_FIXED><<<grid, block, 0, stream>>>(
      reinterpret_cast<const T*>(x.data_ptr()),
      reinterpret_cast<const T*>(centroids.data_ptr()),
      c_sq.data_ptr<float>(),
      cluster_ids.data_ptr<int32_t>(),
      N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T, int D_FIXED, int TILE_N = SMALL_D_BLOCK_N>
void launch_assign_small_d_tiled(
    const at::Tensor& x,
    const at::Tensor& centroids,
    const at::Tensor& c_sq,
    at::Tensor& cluster_ids) {
  int B = x.size(0);
  int N = x.size(1);
  int K = centroids.size(1);

  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid((N + TILE_N - 1) / TILE_N, B);
  dim3 block(TILE_N);
  size_t smem = (size_t)SMALL_D_BLOCK_K * D_FIXED * sizeof(T) +
      (size_t)SMALL_D_BLOCK_K * sizeof(float);
  assign_small_d_tiled_kernel<T, D_FIXED, TILE_N, SMALL_D_BLOCK_K><<<grid, block, smem, stream>>>(
      reinterpret_cast<const T*>(x.data_ptr()),
      reinterpret_cast<const T*>(centroids.data_ptr()),
      c_sq.data_ptr<float>(),
      cluster_ids.data_ptr<int32_t>(),
      N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
__global__ void __launch_bounds__(D1_SORT_THREADS, 4)
assign_d1_sorted_kernel(
    const T* __restrict__ x,             // (B, N, 1)
    const float* __restrict__ c_sorted,  // (B, K), ascending fp32
    const int64_t* __restrict__ c_idx,   // (B, K), original centroid ids
    int32_t* __restrict__ cluster_ids,   // (B, N)
    int N, int K) {
  const int pid_b = blockIdx.y;
  const int n_idx = blockIdx.x * D1_SORT_THREADS + threadIdx.x;
  if (n_idx >= N) return;

  const float xv = to_fp32(x[(size_t)pid_b * N + n_idx]);
  const float* vals = c_sorted + (size_t)pid_b * K;
  const int64_t* idxs = c_idx + (size_t)pid_b * K;

  int lo = 0;
  int hi = K;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (vals[mid] < xv) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  float best_dist = FLT_MAX;
  int best_idx = 0;
  if (lo < K) {
    consider_d1_sorted_group(vals, idxs, K, lo, xv, best_dist, best_idx);
  }
  if (lo > 0) {
    consider_d1_sorted_group(vals, idxs, K, lo - 1, xv, best_dist, best_idx);
  }

  cluster_ids[(size_t)pid_b * N + n_idx] = best_idx;
}

template <typename T>
void launch_assign_d1_sorted_typed(
    const at::Tensor& x,
    const at::Tensor& centroids,
    at::Tensor& cluster_ids) {
  int B = x.size(0);
  int N = x.size(1);
  int K = centroids.size(1);

  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  at::Tensor c_vals = centroids.reshape({B, K}).to(at::kFloat);
  auto sorted = at::sort(c_vals, /*stable=*/true, /*dim=*/1, /*descending=*/false);
  at::Tensor c_sorted = std::get<0>(sorted).contiguous();
  at::Tensor c_idx = std::get<1>(sorted).contiguous();

  dim3 grid((N + D1_SORT_THREADS - 1) / D1_SORT_THREADS, B);
  dim3 block(D1_SORT_THREADS);
  assign_d1_sorted_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const T*>(x.data_ptr()),
      c_sorted.data_ptr<float>(),
      c_idx.data_ptr<int64_t>(),
      cluster_ids.data_ptr<int32_t>(),
      N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
bool try_launch_assign_small_d_typed(
    const at::Tensor& x,
    const at::Tensor& centroids,
    const at::Tensor& c_sq,
    at::Tensor& cluster_ids) {
  switch (static_cast<int>(x.size(2))) {
    case  1: launch_assign_small_d_tiled<T,  1>(x, centroids, c_sq, cluster_ids); return true;
    case  2: launch_assign_small_d_tiled<T,  2>(x, centroids, c_sq, cluster_ids); return true;
    case  3: launch_assign_small_d_tiled<T,  3>(x, centroids, c_sq, cluster_ids); return true;
    case  4: launch_assign_small_d_tiled<T,  4>(x, centroids, c_sq, cluster_ids); return true;
    case  5: launch_assign_small_d_tiled<T,  5>(x, centroids, c_sq, cluster_ids); return true;
    case  6: launch_assign_small_d_tiled<T,  6>(x, centroids, c_sq, cluster_ids); return true;
    case  7: launch_assign_small_d_tiled<T,  7>(x, centroids, c_sq, cluster_ids); return true;
    case  8: launch_assign_small_d_fixed<T,  8>(x, centroids, c_sq, cluster_ids); return true;
    case  9: launch_assign_small_d_fixed<T,  9>(x, centroids, c_sq, cluster_ids); return true;
    case 10: launch_assign_small_d_fixed<T, 10>(x, centroids, c_sq, cluster_ids); return true;
    case 11: launch_assign_small_d_fixed<T, 11>(x, centroids, c_sq, cluster_ids); return true;
    case 12: launch_assign_small_d_fixed<T, 12>(x, centroids, c_sq, cluster_ids); return true;
    case 13: launch_assign_small_d_fixed<T, 13>(x, centroids, c_sq, cluster_ids); return true;
    case 14: launch_assign_small_d_fixed<T, 14>(x, centroids, c_sq, cluster_ids); return true;
    case 15: launch_assign_small_d_fixed<T, 15>(x, centroids, c_sq, cluster_ids); return true;
    default: return false;
  }
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
    if (x.size(2) == 1) {
      launch_assign_d1_sorted_typed<__half>(x, centroids, cluster_ids);
      return;
    }
    if (try_launch_assign_small_d_typed<__half>(x, centroids, c_sq, cluster_ids)) return;
    launch_assign_safe_typed<__half>(x, centroids, x_sq, c_sq, cluster_ids);
  } else if (x.scalar_type() == at::kBFloat16) {
    if (x.size(2) == 1) {
      launch_assign_d1_sorted_typed<__nv_bfloat16>(x, centroids, cluster_ids);
      return;
    }
    if (try_launch_assign_small_d_typed<__nv_bfloat16>(x, centroids, c_sq, cluster_ids)) return;
    launch_assign_safe_typed<__nv_bfloat16>(x, centroids, x_sq, c_sq, cluster_ids);
  } else if (x.scalar_type() == at::kFloat) {
    if (try_launch_assign_small_d_typed<float>(x, centroids, c_sq, cluster_ids)) return;
    launch_assign_safe_typed<float>(x, centroids, x_sq, c_sq, cluster_ids);
  } else {
    TORCH_CHECK(false, "assign_safe: unsupported dtype ", x.scalar_type());
  }
}

}  // namespace assign
}  // namespace fkc
