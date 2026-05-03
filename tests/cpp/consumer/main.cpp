#include <iostream>
#include <stdexcept>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <flash_kmeans_cuda/flash_kmeans_cuda.h>

namespace {

void check(bool ok, const char* msg) {
  if (!ok) {
    throw std::runtime_error(msg);
  }
}

}  // namespace

int main() {
  try {
    check(at::cuda::is_available(), "CUDA is not available");
    c10::cuda::CUDAGuard guard(0);
    at::manual_seed(0);

    constexpr int64_t B = 1;
    constexpr int64_t N = 512;
    constexpr int64_t K = 64;
    constexpr int64_t D = 64;

    auto opts = at::TensorOptions().device(at::kCUDA).dtype(at::kHalf);
    at::Tensor x = at::randn({B, N, D}, opts).contiguous();
    at::Tensor centroids = at::randn({B, K, D}, opts).contiguous();
    at::Tensor x_sq = x.to(at::kFloat).pow(2).sum(-1).contiguous();
    at::Tensor c_sq = centroids.to(at::kFloat).pow(2).sum(-1).contiguous();

    at::Tensor ids = fkc::euclid_assign(x, centroids, x_sq, c_sq);
    at::cuda::getCurrentCUDAStream().synchronize();

    check(ids.sizes() == at::IntArrayRef({B, N}), "assign output shape mismatch");
    check(ids.scalar_type() == at::kInt, "assign output dtype mismatch");
    std::cout << "consumer smoke ids[0,0]=" << ids[0][0].item<int32_t>() << "\n";
    std::cout << "flash_kmeans_cuda installed C++ consumer smoke passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "flash_kmeans_cuda installed C++ consumer smoke failed: "
              << e.what() << "\n";
    return 1;
  }
}
