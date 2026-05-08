#pragma once

// Public C++ API for linking flash-kmeans-cuda into non-Python projects.
//
// This API is intentionally libtorch-based: callers pass at::Tensor objects and
// link against the same Torch/CUDA runtime as the shared library. The lower
// level kernel launchers remain in assign/assign.h and update/update.h.

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

#if defined(_WIN32)
  #if defined(FKC_BUILD_SHARED)
    #define FKC_API __declspec(dllexport)
  #else
    #define FKC_API __declspec(dllimport)
  #endif
#else
  #define FKC_API __attribute__((visibility("default")))
#endif

namespace fkc {

// Allocate and return (B, N) int32 cluster ids.
FKC_API at::Tensor euclid_assign(
    const at::Tensor& x,
    const at::Tensor& centroids,
    const at::Tensor& x_sq,
    const at::Tensor& c_sq);

// Write into caller-provided (B, N) int32 cluster_ids and return it.
FKC_API at::Tensor euclid_assign_out(
    const at::Tensor& x,
    const at::Tensor& centroids,
    const at::Tensor& x_sq,
    const at::Tensor& c_sq,
    at::Tensor cluster_ids);

// Allocate and return (B, N) int32 ids for argmax_k x @ centroid_k.
FKC_API at::Tensor similarity_assign(
    const at::Tensor& x,
    const at::Tensor& centroids);

// Write similarity assignment into caller-provided (B, N) int32 cluster_ids.
FKC_API at::Tensor similarity_assign_out(
    const at::Tensor& x,
    const at::Tensor& centroids,
    at::Tensor cluster_ids);

// Sorted-run centroid accumulation. centroid_sums and centroid_counts are
// zeroed by this wrapper before launching the CUDA accumulator.
FKC_API void centroid_update_sorted(
    const at::Tensor& x_sorted,
    const at::Tensor& cluster_ids_sorted,
    at::Tensor centroid_sums,
    at::Tensor centroid_counts);

// Same accumulator, but reads rows from original-order x via sorted_idx.
FKC_API void centroid_update_sorted_indexed(
    const at::Tensor& x,
    const at::Tensor& sorted_idx,
    const at::Tensor& cluster_ids_sorted,
    at::Tensor centroid_sums,
    at::Tensor centroid_counts);

// Allocate and return finalized centroids.
FKC_API at::Tensor centroid_finalize(
    const at::Tensor& centroid_sums,
    const at::Tensor& centroid_counts,
    const at::Tensor& old_centroids);

// Write finalized centroids into caller-provided out and return it.
FKC_API at::Tensor centroid_finalize_out(
    const at::Tensor& centroid_sums,
    const at::Tensor& centroid_counts,
    const at::Tensor& old_centroids,
    at::Tensor out);

}  // namespace fkc
