#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "flash_kmeans_cuda.h"

namespace {

void check(bool ok, const char* msg) {
  if (!ok) {
    throw std::runtime_error(msg);
  }
}

at::Tensor reference_assign(
    const at::Tensor& x,
    const at::Tensor& centroids,
    const at::Tensor& x_sq,
    const at::Tensor& c_sq) {
  at::Tensor cross = at::matmul(x.to(at::kFloat), centroids.to(at::kFloat).transpose(1, 2));
  at::Tensor dist = x_sq.unsqueeze(-1) + c_sq.unsqueeze(1) - 2.0 * cross;
  return std::get<1>(dist.min(-1)).to(at::kInt);
}

void test_assign(int64_t D) {
  constexpr int64_t B = 1;
  constexpr int64_t N = 512;
  constexpr int64_t K = 64;

  auto opts = at::TensorOptions().device(at::kCUDA).dtype(at::kHalf);
  at::Tensor x = at::randn({B, N, D}, opts).contiguous();
  at::Tensor centroids = at::randn({B, K, D}, opts).contiguous();
  at::Tensor x_sq = x.to(at::kFloat).pow(2).sum(-1).contiguous();
  at::Tensor c_sq = centroids.to(at::kFloat).pow(2).sum(-1).contiguous();

  at::Tensor ids = fkc::euclid_assign(x, centroids, x_sq, c_sq);
  at::cuda::getCurrentCUDAStream().synchronize();

  at::Tensor ref = reference_assign(x, centroids, x_sq, c_sq);
  at::cuda::getCurrentCUDAStream().synchronize();

  check(ids.sizes() == at::IntArrayRef({B, N}), "assign output shape mismatch");
  check(ids.scalar_type() == at::kInt, "assign output dtype mismatch");
  const double disagreement = ids.ne(ref).to(at::kFloat).mean().item<double>();
  std::cout << "assign D=" << D << " disagreement=" << disagreement << "\n";
  check(disagreement < 0.05, "assign disagreement exceeds 5%");
}

void test_update_finalize(int64_t D) {
  constexpr int64_t B = 1;
  constexpr int64_t N = 512;
  constexpr int64_t K = 16;

  auto x_opts = at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
  auto i32_opts = at::TensorOptions().device(at::kCUDA).dtype(at::kInt);
  at::Tensor x = at::randn({B, N, D}, x_opts).contiguous();
  at::Tensor ids = at::randint(K, {B, N}, i32_opts).contiguous();

  auto sorted_pair = at::sort(ids, 1);
  at::Tensor sorted_ids = std::get<0>(sorted_pair).contiguous();
  at::Tensor sorted_idx = std::get<1>(sorted_pair).to(at::kInt).contiguous();
  at::Tensor x_sorted = x.gather(
      1, sorted_idx.to(at::kLong).unsqueeze(-1).expand({B, N, D})).contiguous();

  at::Tensor sums = at::empty({B, K, D}, x_opts);
  at::Tensor counts = at::empty({B, K}, i32_opts);
  fkc::centroid_update_sorted(x_sorted, sorted_ids, sums, counts);
  at::cuda::getCurrentCUDAStream().synchronize();

  at::Tensor sums_indexed = at::empty_like(sums);
  at::Tensor counts_indexed = at::empty_like(counts);
  fkc::centroid_update_sorted_indexed(x, sorted_idx, sorted_ids, sums_indexed, counts_indexed);
  at::cuda::getCurrentCUDAStream().synchronize();

  check(at::allclose(sums, sums_indexed, 1e-4, 1e-4), "indexed update sums mismatch");
  check(counts.equal(counts_indexed), "indexed update counts mismatch");

  at::Tensor old_centroids = at::randn({B, K, D}, x_opts).contiguous();
  at::Tensor finalized = fkc::centroid_finalize(sums, counts, old_centroids);
  at::cuda::getCurrentCUDAStream().synchronize();
  check(finalized.sizes() == at::IntArrayRef({B, K, D}), "finalize output shape mismatch");

  at::Tensor ref = at::where(
      counts.unsqueeze(-1).gt(0),
      sums / counts.clamp_min(1).to(at::kFloat).unsqueeze(-1),
      old_centroids);
  check(at::allclose(finalized, ref, 1e-4, 1e-4), "finalize output mismatch");
  std::cout << "update/finalize D=" << D << " passed\n";
}

}  // namespace

int main() {
  try {
    check(at::cuda::is_available(), "CUDA is not available");
    c10::cuda::CUDAGuard guard(0);
    at::manual_seed(0);
    for (int64_t D : std::vector<int64_t>{1, 7, 16, 31, 32, 48, 64, 96, 128, 256, 257}) {
      test_assign(D);
    }
    for (int64_t D : std::vector<int64_t>{1, 7, 16, 31, 32, 64, 128, 257}) {
      test_update_finalize(D);
    }
    std::cout << "flash_kmeans_cuda C++ smoke test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "flash_kmeans_cuda C++ smoke test failed: " << e.what() << "\n";
    return 1;
  }
}
