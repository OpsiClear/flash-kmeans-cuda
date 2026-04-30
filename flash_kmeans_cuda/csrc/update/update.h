#pragma once

// See assign.h — kernel TUs use <ATen/ATen.h> instead of torch headers to
// avoid Python.h and torch/csrc/dynamo/compiled_autograd.h.
#ifdef small
  #undef small
#endif
#ifdef min
  #undef min
#endif
#ifdef max
  #undef max
#endif

#include <ATen/ATen.h>

namespace fkc {
namespace update {

// Sorted-chunk centroid accumulator. Caller must pre-sort cluster_ids along N
// and gather x rows accordingly (see kmeans.py — the host does torch.sort).
//
// Inputs:
//   x_sorted        : (B, N, D) compute dtype, contiguous, sorted along N
//   cluster_ids_sorted: (B, N)  int32, contiguous, monotonically non-decreasing per batch
//   centroid_sums   : (B, K, D) fp32, contiguous, *zero-initialized by caller*
//   centroid_counts : (B, K)    int32, contiguous, *zero-initialized by caller*
void launch_centroid_update_sorted(
    const at::Tensor& x_sorted,
    const at::Tensor& cluster_ids_sorted,
    at::Tensor& centroid_sums,
    at::Tensor& centroid_counts);

// Finalize: new_centroids = where(count > 0, sums / count, old_centroids)
// then cast to old_centroids.dtype.
//
// Inputs:
//   centroid_sums   : (B, K, D) fp32
//   centroid_counts : (B, K)    int32
//   old_centroids   : (B, K, D) compute dtype
// Output:
//   new_centroids   : (B, K, D) compute dtype (allocated by caller)
void launch_centroid_finalize(
    const at::Tensor& centroid_sums,
    const at::Tensor& centroid_counts,
    const at::Tensor& old_centroids,
    at::Tensor& new_centroids);

}  // namespace update
}  // namespace fkc
