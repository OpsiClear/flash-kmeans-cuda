// Sorted-chunk centroid accumulator.
//
// Caller has pre-sorted cluster_ids[b, :] along N and gathered x rows in the
// same permutation. Each CTA processes BLOCK_N consecutive sorted tokens; the
// cluster_ids form contiguous runs, so we accumulate features per run in
// registers and emit ONE atomicAdd per run-and-feature-chunk to
// centroid_sums[b, cid, :] (vs one atomic-per-token in Triton's atomic-add
// variant). counts[b, cid] gets one atomicAdd per run.
//
// Algorithm:
//   1. Each warp owns a contiguous slice of the BLOCK_N tokens.
//   2. Walk the slice token-by-token, comparing cluster_ids[i] vs cluster_ids[i-1].
//   3. When we hit a boundary, flush the accumulator to global with one atomic
//      add per BLOCK_D chunk of features, plus one atomicAdd to count.
//   4. Runs that span CTA boundaries simply have each CTA atomicAdd into the
//      same destination — atomically correct without coordination.

#include "update.h"
#include "../common/arch.cuh"

#include "../common/torch_cuda_includes.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace fkc {
namespace update {

namespace {

constexpr int BLOCK_N = 256;          // tokens per CTA
constexpr int THREADS_PER_CTA = 128;

// Convert a single fp16/bf16 element to fp32.
__device__ __forceinline__ float to_fp32(__half h) { return __half2float(h); }
__device__ __forceinline__ float to_fp32(__nv_bfloat16 b) { return __bfloat162float(b); }
__device__ __forceinline__ float to_fp32(float f) { return f; }


template <typename T>
__global__ void __launch_bounds__(THREADS_PER_CTA, 4)
update_sorted_kernel(
    const T* __restrict__ x,                // (B, N, D)
    const int32_t* __restrict__ cluster_ids,// (B, N)
    float* __restrict__ centroid_sums,      // (B, K, D)  fp32
    int32_t* __restrict__ centroid_counts,  // (B, K)     int32
    int B, int N, int K, int D) {
  const int pid_n = blockIdx.x;
  const int pid_b = blockIdx.y;
  const int tid = threadIdx.x;

  const int n_start = pid_n * BLOCK_N;
  const int n_count = min(BLOCK_N, N - n_start);
  if (n_count <= 0) return;

  const int32_t* ids = cluster_ids + (size_t)pid_b * N;
  const T* xb = x + (size_t)pid_b * N * D;
  float* sumb = centroid_sums + (size_t)pid_b * K * D;
  int32_t* cntb = centroid_counts + (size_t)pid_b * K;

  // Cooperatively walk the slice. Strategy:
  //   - thread 0 of each warp finds the run boundaries within its slice.
  //   - All threads of the warp cooperate on flushing each run's feature sum
  //     to global via atomicAdd, with each thread owning a strided slice of D.
  //
  // Simpler implementation: one warp owns the entire BLOCK_N slice (64
  // tokens for warp_id == 0 ... warp_id == 3 takes 64..255). Within each
  // warp's slice we identify runs serially in lane 0 and broadcast.
  //
  // For maximum throughput we instead let the entire CTA (all 128 threads)
  // cooperate per run: lane 0 finds the next run, broadcasts (start, len, cid),
  // then all 128 threads flush the run's feature accumulator in parallel.

  __shared__ int32_t s_run_cid;
  __shared__ int s_run_start;
  __shared__ int s_run_len;

  int cursor = 0;
  while (cursor < n_count) {
    if (tid == 0) {
      int32_t cid = ids[n_start + cursor];
      int run_end = cursor + 1;
      while (run_end < n_count && ids[n_start + run_end] == cid) ++run_end;
      s_run_cid = cid;
      s_run_start = cursor;
      s_run_len = run_end - cursor;
    }
    __syncthreads();

    int32_t cid = s_run_cid;
    int run_start = s_run_start;
    int run_len = s_run_len;

    // Bounds-check cid (should always be in [0, K), but Triton kernel does
    // a similar guard in centroid_update_triton.py:41).
    if (cid >= 0 && cid < K) {
      // Each thread owns a feature index strided by THREADS_PER_CTA.
      for (int d = tid; d < D; d += THREADS_PER_CTA) {
        float acc = 0.f;
        // Sum the run's features at column d.
        #pragma unroll 4
        for (int r = 0; r < run_len; ++r) {
          T v = xb[(size_t)(n_start + run_start + r) * D + d];
          acc += to_fp32(v);
        }
        atomicAdd(sumb + (size_t)cid * D + d, acc);
      }
      // One thread updates the count.
      if (tid == 0) {
        atomicAdd(cntb + cid, run_len);
      }
    }

    cursor = run_start + run_len;
    __syncthreads();
  }
}

template <typename T>
__global__ void __launch_bounds__(THREADS_PER_CTA, 4)
update_sorted_indexed_kernel(
    const T* __restrict__ x,                // (B, N, D), original order
    const int32_t* __restrict__ sorted_idx, // (B, N), sorted permutation
    const int32_t* __restrict__ cluster_ids,// (B, N), sorted cluster ids
    float* __restrict__ centroid_sums,      // (B, K, D)  fp32
    int32_t* __restrict__ centroid_counts,  // (B, K)     int32
    int B, int N, int K, int D) {
  const int pid_n = blockIdx.x;
  const int pid_b = blockIdx.y;
  const int tid = threadIdx.x;

  const int n_start = pid_n * BLOCK_N;
  const int n_count = min(BLOCK_N, N - n_start);
  if (n_count <= 0) return;

  const int32_t* ids = cluster_ids + (size_t)pid_b * N;
  const int32_t* idx = sorted_idx + (size_t)pid_b * N;
  const T* xb = x + (size_t)pid_b * N * D;
  float* sumb = centroid_sums + (size_t)pid_b * K * D;
  int32_t* cntb = centroid_counts + (size_t)pid_b * K;

  __shared__ int32_t s_run_cid;
  __shared__ int s_run_start;
  __shared__ int s_run_len;

  int cursor = 0;
  while (cursor < n_count) {
    if (tid == 0) {
      int32_t cid = ids[n_start + cursor];
      int run_end = cursor + 1;
      while (run_end < n_count && ids[n_start + run_end] == cid) ++run_end;
      s_run_cid = cid;
      s_run_start = cursor;
      s_run_len = run_end - cursor;
    }
    __syncthreads();

    int32_t cid = s_run_cid;
    int run_start = s_run_start;
    int run_len = s_run_len;

    if (cid >= 0 && cid < K) {
      for (int d = tid; d < D; d += THREADS_PER_CTA) {
        float acc = 0.f;
        #pragma unroll 4
        for (int r = 0; r < run_len; ++r) {
          int row = idx[n_start + run_start + r];
          T v = xb[(size_t)row * D + d];
          acc += to_fp32(v);
        }
        atomicAdd(sumb + (size_t)cid * D + d, acc);
      }
      if (tid == 0) {
        atomicAdd(cntb + cid, run_len);
      }
    }

    cursor = run_start + run_len;
    __syncthreads();
  }
}

}  // namespace


void launch_centroid_update_sorted(
    const at::Tensor& x_sorted,
    const at::Tensor& cluster_ids_sorted,
    at::Tensor& centroid_sums,
    at::Tensor& centroid_counts) {
  TORCH_CHECK(x_sorted.is_cuda(), "x_sorted must be CUDA");
  TORCH_CHECK(x_sorted.dim() == 3, "x_sorted must be (B, N, D)");
  TORCH_CHECK(cluster_ids_sorted.scalar_type() == at::kInt,
              "cluster_ids must be int32");
  TORCH_CHECK(centroid_sums.scalar_type() == at::kFloat,
              "centroid_sums must be fp32");
  TORCH_CHECK(centroid_counts.scalar_type() == at::kInt,
              "centroid_counts must be int32");
  TORCH_CHECK(x_sorted.is_contiguous() && cluster_ids_sorted.is_contiguous() &&
              centroid_sums.is_contiguous() && centroid_counts.is_contiguous(),
              "all tensors must be contiguous");

  int B = x_sorted.size(0);
  int N = x_sorted.size(1);
  int D = x_sorted.size(2);
  int K = centroid_sums.size(1);

  c10::cuda::CUDAGuard guard(x_sorted.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid((N + BLOCK_N - 1) / BLOCK_N, B);
  dim3 block(THREADS_PER_CTA);

  if (x_sorted.scalar_type() == at::kHalf) {
    update_sorted_kernel<__half><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(x_sorted.data_ptr()),
        cluster_ids_sorted.data_ptr<int32_t>(),
        centroid_sums.data_ptr<float>(),
        centroid_counts.data_ptr<int32_t>(),
        B, N, K, D);
  } else if (x_sorted.scalar_type() == at::kBFloat16) {
    update_sorted_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_sorted.data_ptr()),
        cluster_ids_sorted.data_ptr<int32_t>(),
        centroid_sums.data_ptr<float>(),
        centroid_counts.data_ptr<int32_t>(),
        B, N, K, D);
  } else if (x_sorted.scalar_type() == at::kFloat) {
    update_sorted_kernel<float><<<grid, block, 0, stream>>>(
        x_sorted.data_ptr<float>(),
        cluster_ids_sorted.data_ptr<int32_t>(),
        centroid_sums.data_ptr<float>(),
        centroid_counts.data_ptr<int32_t>(),
        B, N, K, D);
  } else {
    TORCH_CHECK(false, "centroid_update_sorted: unsupported dtype ",
                x_sorted.scalar_type());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_centroid_update_sorted_indexed(
    const at::Tensor& x,
    const at::Tensor& sorted_idx,
    const at::Tensor& cluster_ids_sorted,
    at::Tensor& centroid_sums,
    at::Tensor& centroid_counts) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(x.dim() == 3, "x must be (B, N, D)");
  TORCH_CHECK(sorted_idx.scalar_type() == at::kInt,
              "sorted_idx must be int32");
  TORCH_CHECK(cluster_ids_sorted.scalar_type() == at::kInt,
              "cluster_ids must be int32");
  TORCH_CHECK(centroid_sums.scalar_type() == at::kFloat,
              "centroid_sums must be fp32");
  TORCH_CHECK(centroid_counts.scalar_type() == at::kInt,
              "centroid_counts must be int32");
  TORCH_CHECK(x.is_contiguous() && sorted_idx.is_contiguous() &&
              cluster_ids_sorted.is_contiguous() && centroid_sums.is_contiguous() &&
              centroid_counts.is_contiguous(),
              "all tensors must be contiguous");

  int B = x.size(0);
  int N = x.size(1);
  int D = x.size(2);
  int K = centroid_sums.size(1);
  TORCH_CHECK(sorted_idx.dim() == 2 && sorted_idx.size(0) == B &&
              sorted_idx.size(1) == N,
              "sorted_idx must be (B, N)");
  TORCH_CHECK(cluster_ids_sorted.dim() == 2 && cluster_ids_sorted.size(0) == B &&
              cluster_ids_sorted.size(1) == N,
              "cluster_ids_sorted must be (B, N)");

  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid((N + BLOCK_N - 1) / BLOCK_N, B);
  dim3 block(THREADS_PER_CTA);

  if (x.scalar_type() == at::kHalf) {
    update_sorted_indexed_kernel<__half><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr()),
        sorted_idx.data_ptr<int32_t>(),
        cluster_ids_sorted.data_ptr<int32_t>(),
        centroid_sums.data_ptr<float>(),
        centroid_counts.data_ptr<int32_t>(),
        B, N, K, D);
  } else if (x.scalar_type() == at::kBFloat16) {
    update_sorted_indexed_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        sorted_idx.data_ptr<int32_t>(),
        cluster_ids_sorted.data_ptr<int32_t>(),
        centroid_sums.data_ptr<float>(),
        centroid_counts.data_ptr<int32_t>(),
        B, N, K, D);
  } else if (x.scalar_type() == at::kFloat) {
    update_sorted_indexed_kernel<float><<<grid, block, 0, stream>>>(
        x.data_ptr<float>(),
        sorted_idx.data_ptr<int32_t>(),
        cluster_ids_sorted.data_ptr<int32_t>(),
        centroid_sums.data_ptr<float>(),
        centroid_counts.data_ptr<int32_t>(),
        B, N, K, D);
  } else {
    TORCH_CHECK(false, "centroid_update_sorted_indexed: unsupported dtype ",
                x.scalar_type());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


}  // namespace update
}  // namespace fkc
