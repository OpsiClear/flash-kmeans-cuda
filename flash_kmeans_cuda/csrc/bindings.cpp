// nanobind entry point for flash_kmeans_cuda._C.
//
// Exposes assignment/update ops used by the Python package.
// Tensor handoff between Python and C++ uses the at::Tensor caster in
// ``nb_torch.h`` (nanobind doesn't ship one).
//
// Dispatch:
//   euclid_assign/similarity_assign route fp16/bf16 to sm_80 mma.sync kernels
//   by default; fp32 falls back to safe kernels.
//   Set FKC_ASSIGN_FORCE_SAFE=1 to override and use the safe kernel for all
//   dtypes (useful for A/B-correctness debugging).

#include <nanobind/nanobind.h>
#include "nb_torch.h"

#include "flash_kmeans_cuda.h"
#include "update/update.h"

namespace nb = nanobind;

namespace {

at::Tensor euclid_assign(
    at::Tensor x,
    at::Tensor centroids,
    at::Tensor x_sq,
    at::Tensor c_sq,
    c10::optional<at::Tensor> out) {
  if (out.has_value()) {
    return fkc::euclid_assign_out(x, centroids, x_sq, c_sq, out.value());
  }
  return fkc::euclid_assign(x, centroids, x_sq, c_sq);
}

at::Tensor similarity_assign(
    at::Tensor x,
    at::Tensor centroids,
    c10::optional<at::Tensor> out) {
  if (out.has_value()) {
    return fkc::similarity_assign_out(x, centroids, out.value());
  }
  return fkc::similarity_assign(x, centroids);
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
  m.def("similarity_assign", &similarity_assign,
        nb::arg("x"), nb::arg("centroids"), nb::arg("out") = nb::none(),
        "Compute argmax_k x @ c_k for each point. Returns (B,N) int32.");
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
