// nanobind entry point for flash_kmeans_cuda._C.
//
// Exposes three ops: euclid_assign, centroid_update_sorted, centroid_finalize.
// Tensor handoff between Python and C++ uses the at::Tensor caster in
// ``nb_torch.h`` (nanobind doesn't ship one).
//
// Dispatch:
//   euclid_assign routes fp16/bf16 to the sm_80 mma.sync kernel by default;
//   fp32 (no fp32-input mma path on sm_80) falls back to the safe kernel.
//   Set FKC_ASSIGN_FORCE_SAFE=1 to override and use the safe kernel for all
//   dtypes (useful for A/B-correctness debugging).

#include <nanobind/nanobind.h>
#include "nb_torch.h"
#include <c10/cuda/CUDAStream.h>
#include <cstdlib>
#include <cstring>

#include "assign/assign.h"
#include "update/update.h"

namespace nb = nanobind;

namespace {

// Opt-out flag: force the safe (tensor-core-free) kernel for every dtype.
// Used for A/B correctness debugging.
bool force_safe_path() {
  static const bool v = []() {
    const char* s = std::getenv("FKC_ASSIGN_FORCE_SAFE");
    if (!s) return false;
    return std::strcmp(s, "1") == 0 || std::strcmp(s, "true") == 0;
  }();
  return v;
}

at::Tensor euclid_assign(
    at::Tensor x,
    at::Tensor centroids,
    at::Tensor x_sq,
    at::Tensor c_sq,
    c10::optional<at::Tensor> out) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  int B = x.size(0);
  int N = x.size(1);

  at::Tensor cluster_ids;
  if (out.has_value()) {
    cluster_ids = out.value();
    TORCH_CHECK(cluster_ids.scalar_type() == at::kInt, "out must be int32");
    TORCH_CHECK(cluster_ids.dim() == 2 && cluster_ids.size(0) == B &&
                cluster_ids.size(1) == N,
                "out must be (B, N) int32");
  } else {
    cluster_ids = at::empty({B, N},
        at::TensorOptions().dtype(at::kInt).device(x.device()));
  }

  auto dtype = x.scalar_type();
  const bool tc_eligible =
      (dtype == at::kHalf || dtype == at::kBFloat16);
  if (tc_eligible && !force_safe_path()) {
    fkc::assign::launch_assign_sm80(x, centroids, x_sq, c_sq, cluster_ids);
  } else {
    fkc::assign::launch_assign_safe(x, centroids, x_sq, c_sq, cluster_ids);
  }
  return cluster_ids;
}

void centroid_update_sorted(
    at::Tensor x_sorted,
    at::Tensor cluster_ids_sorted,
    at::Tensor centroid_sums,
    at::Tensor centroid_counts) {
  fkc::update::launch_centroid_update_sorted(
      x_sorted, cluster_ids_sorted, centroid_sums, centroid_counts);
}

void centroid_update_sorted_indexed(
    at::Tensor x,
    at::Tensor sorted_idx,
    at::Tensor cluster_ids_sorted,
    at::Tensor centroid_sums,
    at::Tensor centroid_counts) {
  fkc::update::launch_centroid_update_sorted_indexed(
      x, sorted_idx, cluster_ids_sorted, centroid_sums, centroid_counts);
}

at::Tensor centroid_finalize(
    at::Tensor centroid_sums,
    at::Tensor centroid_counts,
    at::Tensor old_centroids,
    c10::optional<at::Tensor> out) {
  at::Tensor new_centroids =
      out.has_value() ? out.value() : at::empty_like(old_centroids);
  fkc::update::launch_centroid_finalize(
      centroid_sums, centroid_counts, old_centroids, new_centroids);
  return new_centroids;
}

}  // namespace


NB_MODULE(_C, m) {
  m.doc() = "flash_kmeans_cuda - hand-rolled CUDA kernels for batched Euclidean K-Means";

  m.def("euclid_assign", &euclid_assign,
        nb::arg("x"), nb::arg("centroids"), nb::arg("x_sq"), nb::arg("c_sq"),
        nb::arg("out") = nb::none(),
        "Compute argmin_k ||x - c_k||^2 for each point. Returns (B,N) int32.");
  m.def("centroid_update_sorted", &centroid_update_sorted,
        nb::arg("x_sorted"), nb::arg("cluster_ids_sorted"),
        nb::arg("centroid_sums"), nb::arg("centroid_counts"),
        "Sorted-chunk centroid sum/count accumulator (in-place into sums/counts).");
  m.def("centroid_update_sorted_indexed", &centroid_update_sorted_indexed,
        nb::arg("x"), nb::arg("sorted_idx"), nb::arg("cluster_ids_sorted"),
        nb::arg("centroid_sums"), nb::arg("centroid_counts"),
        "Sorted-chunk centroid accumulator using original x plus sorted indices.");
  m.def("centroid_finalize", &centroid_finalize,
        nb::arg("centroid_sums"), nb::arg("centroid_counts"),
        nb::arg("old_centroids"),
        nb::arg("out") = nb::none(),
        "sums/counts -> new centroids; keep old for empty clusters.");
}
