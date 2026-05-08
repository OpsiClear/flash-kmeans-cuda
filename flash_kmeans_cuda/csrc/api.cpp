#include "flash_kmeans_cuda.h"

#include <cstdlib>
#include <cstring>

#include "assign/assign.h"
#include "update/update.h"

namespace fkc {
namespace {

bool force_safe_path() {
  const char* s = std::getenv("FKC_ASSIGN_FORCE_SAFE");
  if (!s) return false;
  return std::strcmp(s, "1") == 0 || std::strcmp(s, "true") == 0;
}

bool can_use_sm80_path(const at::Tensor& x) {
  const auto dtype = x.scalar_type();
  const int64_t D = x.size(2);
  return (dtype == at::kHalf || dtype == at::kBFloat16) &&
      ((D >= 3 && D < 16) || (D % 16) == 0);
}

void validate_assign_inputs(
    const at::Tensor& x,
    const at::Tensor& centroids,
    const at::Tensor& x_sq,
    const at::Tensor& c_sq,
    const at::Tensor& cluster_ids) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(centroids.is_cuda(), "centroids must be a CUDA tensor");
  TORCH_CHECK(x_sq.is_cuda(), "x_sq must be a CUDA tensor");
  TORCH_CHECK(c_sq.is_cuda(), "c_sq must be a CUDA tensor");
  TORCH_CHECK(cluster_ids.is_cuda(), "cluster_ids must be a CUDA tensor");
  TORCH_CHECK(x.dim() == 3, "x must be (B, N, D)");
  TORCH_CHECK(centroids.dim() == 3, "centroids must be (B, K, D)");
  TORCH_CHECK(x.scalar_type() == centroids.scalar_type(),
              "x and centroids must share dtype");
  TORCH_CHECK(x_sq.scalar_type() == at::kFloat, "x_sq must be fp32");
  TORCH_CHECK(c_sq.scalar_type() == at::kFloat, "c_sq must be fp32");
  TORCH_CHECK(cluster_ids.scalar_type() == at::kInt, "cluster_ids must be int32");

  const int64_t B = x.size(0);
  const int64_t N = x.size(1);
  const int64_t D = x.size(2);
  const int64_t K = centroids.size(1);
  TORCH_CHECK(centroids.size(0) == B && centroids.size(2) == D,
              "centroids must be (B, K, D) matching x");
  TORCH_CHECK(x_sq.dim() == 2 && x_sq.size(0) == B && x_sq.size(1) == N,
              "x_sq must be (B, N) fp32 matching x");
  TORCH_CHECK(c_sq.dim() == 2 && c_sq.size(0) == B && c_sq.size(1) == K,
              "c_sq must be (B, K) fp32 matching centroids");
  TORCH_CHECK(cluster_ids.dim() == 2 && cluster_ids.size(0) == x.size(0) &&
              cluster_ids.size(1) == x.size(1),
              "cluster_ids must be (B, N) int32");
  TORCH_CHECK(x.device() == centroids.device() &&
              x.device() == x_sq.device() &&
              x.device() == c_sq.device() &&
              x.device() == cluster_ids.device(),
              "all tensors must be on the same CUDA device");
  TORCH_CHECK(x.is_contiguous() && centroids.is_contiguous() &&
              x_sq.is_contiguous() && c_sq.is_contiguous() &&
              cluster_ids.is_contiguous(),
              "all tensors must be contiguous");
}

void validate_similarity_inputs(
    const at::Tensor& x,
    const at::Tensor& centroids,
    const at::Tensor& cluster_ids) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(centroids.is_cuda(), "centroids must be a CUDA tensor");
  TORCH_CHECK(cluster_ids.is_cuda(), "cluster_ids must be a CUDA tensor");
  TORCH_CHECK(x.dim() == 3, "x must be (B, N, D)");
  TORCH_CHECK(centroids.dim() == 3, "centroids must be (B, K, D)");
  TORCH_CHECK(x.scalar_type() == centroids.scalar_type(),
              "x and centroids must share dtype");
  TORCH_CHECK(cluster_ids.scalar_type() == at::kInt, "cluster_ids must be int32");

  const int64_t B = x.size(0);
  const int64_t N = x.size(1);
  const int64_t D = x.size(2);
  TORCH_CHECK(centroids.size(0) == B && centroids.size(2) == D,
              "centroids must be (B, K, D) matching x");
  TORCH_CHECK(cluster_ids.dim() == 2 && cluster_ids.size(0) == B &&
              cluster_ids.size(1) == N,
              "cluster_ids must be (B, N) int32");
  TORCH_CHECK(x.device() == centroids.device() &&
              x.device() == cluster_ids.device(),
              "all tensors must be on the same CUDA device");
  TORCH_CHECK(x.is_contiguous() && centroids.is_contiguous() &&
              cluster_ids.is_contiguous(),
              "all tensors must be contiguous");
}

}  // namespace

at::Tensor euclid_assign(
    const at::Tensor& x,
    const at::Tensor& centroids,
    const at::Tensor& x_sq,
    const at::Tensor& c_sq) {
  at::Tensor cluster_ids = at::empty(
      {x.size(0), x.size(1)},
      at::TensorOptions().dtype(at::kInt).device(x.device()));
  return euclid_assign_out(x, centroids, x_sq, c_sq, cluster_ids);
}

at::Tensor euclid_assign_out(
    const at::Tensor& x,
    const at::Tensor& centroids,
    const at::Tensor& x_sq,
    const at::Tensor& c_sq,
    at::Tensor cluster_ids) {
  validate_assign_inputs(x, centroids, x_sq, c_sq, cluster_ids);

  if (can_use_sm80_path(x) && !force_safe_path()) {
    fkc::assign::launch_assign_sm80(x, centroids, x_sq, c_sq, cluster_ids);
  } else {
    fkc::assign::launch_assign_safe(x, centroids, x_sq, c_sq, cluster_ids);
  }
  return cluster_ids;
}

at::Tensor similarity_assign(
    const at::Tensor& x,
    const at::Tensor& centroids) {
  at::Tensor cluster_ids = at::empty(
      {x.size(0), x.size(1)},
      at::TensorOptions().dtype(at::kInt).device(x.device()));
  return similarity_assign_out(x, centroids, cluster_ids);
}

at::Tensor similarity_assign_out(
    const at::Tensor& x,
    const at::Tensor& centroids,
    at::Tensor cluster_ids) {
  validate_similarity_inputs(x, centroids, cluster_ids);

  if (can_use_sm80_path(x) && !force_safe_path()) {
    fkc::assign::launch_similarity_assign_sm80(x, centroids, cluster_ids);
  } else {
    fkc::assign::launch_similarity_assign_safe(x, centroids, cluster_ids);
  }
  return cluster_ids;
}

void centroid_update_sorted(
    const at::Tensor& x_sorted,
    const at::Tensor& cluster_ids_sorted,
    at::Tensor centroid_sums,
    at::Tensor centroid_counts) {
  centroid_sums.zero_();
  centroid_counts.zero_();
  fkc::update::launch_centroid_update_sorted(
      x_sorted, cluster_ids_sorted, centroid_sums, centroid_counts);
}

void centroid_update_sorted_indexed(
    const at::Tensor& x,
    const at::Tensor& sorted_idx,
    const at::Tensor& cluster_ids_sorted,
    at::Tensor centroid_sums,
    at::Tensor centroid_counts) {
  centroid_sums.zero_();
  centroid_counts.zero_();
  fkc::update::launch_centroid_update_sorted_indexed(
      x, sorted_idx, cluster_ids_sorted, centroid_sums, centroid_counts);
}

at::Tensor centroid_finalize(
    const at::Tensor& centroid_sums,
    const at::Tensor& centroid_counts,
    const at::Tensor& old_centroids) {
  at::Tensor out = at::empty_like(old_centroids);
  return centroid_finalize_out(centroid_sums, centroid_counts, old_centroids, out);
}

at::Tensor centroid_finalize_out(
    const at::Tensor& centroid_sums,
    const at::Tensor& centroid_counts,
    const at::Tensor& old_centroids,
    at::Tensor out) {
  fkc::update::launch_centroid_finalize(
      centroid_sums, centroid_counts, old_centroids, out);
  return out;
}

}  // namespace fkc
