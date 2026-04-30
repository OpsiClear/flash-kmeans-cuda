// Centroid finalize: divide sums by counts, keep old centroid for empty
// clusters, cast to compute dtype. Trivial — one thread per (b, k, d).

#include "update.h"
#include "../common/arch.cuh"

#include "../common/torch_cuda_includes.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace fkc {
namespace update {

namespace {

template <typename T>
__device__ __forceinline__ T from_fp32(float x);

template <> __device__ __forceinline__ __half from_fp32<__half>(float x) {
  return __float2half(x);
}
template <> __device__ __forceinline__ __nv_bfloat16 from_fp32<__nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}
template <> __device__ __forceinline__ float from_fp32<float>(float x) { return x; }

template <typename T>
__device__ __forceinline__ float to_fp32_t(T x);

template <> __device__ __forceinline__ float to_fp32_t<__half>(__half x) {
  return __half2float(x);
}
template <> __device__ __forceinline__ float to_fp32_t<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}
template <> __device__ __forceinline__ float to_fp32_t<float>(float x) { return x; }


template <typename T>
__global__ void finalize_kernel(
    const float* __restrict__ sums,
    const int32_t* __restrict__ counts,
    const T* __restrict__ old_centroids,
    T* __restrict__ new_centroids,
    int B, int K, int D) {
  size_t total = (size_t)B * K * D;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  int d = idx % D;
  size_t bk = idx / D;
  int k = bk % K;
  int b = bk / K;
  (void)b; (void)d; (void)k;  // only used implicitly via idx

  size_t cnt_idx = bk;        // (b, k)
  int32_t cnt = counts[cnt_idx];
  T out;
  if (cnt > 0) {
    float v = sums[idx] / static_cast<float>(cnt);
    out = from_fp32<T>(v);
  } else {
    out = old_centroids[idx];
  }
  new_centroids[idx] = out;
}

}  // namespace


void launch_centroid_finalize(
    const at::Tensor& centroid_sums,
    const at::Tensor& centroid_counts,
    const at::Tensor& old_centroids,
    at::Tensor& new_centroids) {
  TORCH_CHECK(centroid_sums.is_cuda(), "all tensors must be CUDA");
  TORCH_CHECK(centroid_sums.scalar_type() == at::kFloat, "sums must be fp32");
  TORCH_CHECK(centroid_counts.scalar_type() == at::kInt, "counts must be int32");
  TORCH_CHECK(old_centroids.scalar_type() == new_centroids.scalar_type(),
              "old and new centroids must share dtype");
  TORCH_CHECK(centroid_sums.is_contiguous() && centroid_counts.is_contiguous() &&
              old_centroids.is_contiguous() && new_centroids.is_contiguous(),
              "all tensors must be contiguous");

  int B = centroid_sums.size(0);
  int K = centroid_sums.size(1);
  int D = centroid_sums.size(2);

  c10::cuda::CUDAGuard guard(centroid_sums.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  size_t total = (size_t)B * K * D;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;

  if (old_centroids.scalar_type() == at::kHalf) {
    finalize_kernel<__half><<<blocks, threads, 0, stream>>>(
        centroid_sums.data_ptr<float>(),
        centroid_counts.data_ptr<int32_t>(),
        reinterpret_cast<const __half*>(old_centroids.data_ptr()),
        reinterpret_cast<__half*>(new_centroids.data_ptr()),
        B, K, D);
  } else if (old_centroids.scalar_type() == at::kBFloat16) {
    finalize_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        centroid_sums.data_ptr<float>(),
        centroid_counts.data_ptr<int32_t>(),
        reinterpret_cast<const __nv_bfloat16*>(old_centroids.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(new_centroids.data_ptr()),
        B, K, D);
  } else if (old_centroids.scalar_type() == at::kFloat) {
    finalize_kernel<float><<<blocks, threads, 0, stream>>>(
        centroid_sums.data_ptr<float>(),
        centroid_counts.data_ptr<int32_t>(),
        old_centroids.data_ptr<float>(),
        new_centroids.data_ptr<float>(),
        B, K, D);
  } else {
    TORCH_CHECK(false, "centroid_finalize: unsupported dtype");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace update
}  // namespace fkc
