// flash_kmeans_cuda/csrc/assign/assign_kernel_launch.h
#pragma once

// Template-launch helpers for the sm80 assign kernel. Lives in a header so
// the variant catalog (assign_policy.cu) can reach them without forcing every
// dispatch entry through assign_sm80.cu's translation unit.

#include "assign.h"
#include "assign_common.cuh"
#include "../common/arch.cuh"
#include "../common/torch_cuda_includes.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace fkc {
namespace assign {

// SMEM row-stride padding (in fp16 elements). Shared between the kernel
// (assign_sm80.cu) and the host-side launcher here.
// Adding 8 fp16 = 16 bytes preserves cp.async's required 16-byte alignment
// of the destination SMEM offset for every row, while shifting each subsequent
// SMEM row by 4 banks. Without padding, power-of-2 D (64/128/256) causes
// 32-way bank conflicts across rows; with this padding, the 8 cooperating
// row-bank offsets land on distinct 4-byte banks.
constexpr int SMEM_PAD = 8;

// Forward decl of the kernel template defined in assign_sm80.cu.
// __launch_bounds__ and __restrict__ must match the definition exactly.
template <typename T, int BLOCK_N_, int BLOCK_K_, int WARPS_, int STAGES_,
          int N_TILES_, bool ASYNC_CSQ_, int D_FIXED_, bool RAW_DIST_,
          bool FP32_ACC_ = false, bool SIMILARITY_ = false,
          int SMEM_PAD_ = SMEM_PAD>
__global__ void __launch_bounds__(WARPS_ * 32, 1)
assign_sm80_kernel(const T* __restrict__, const T* __restrict__,
                   const float* __restrict__, const float* __restrict__,
                   int32_t* __restrict__, int, int, int, int);

inline size_t compute_smem_bytes(int BLOCK_N_, int BLOCK_K_, int D,
                                 int PIPE_STAGES_, size_t elt_sz,
                                 int smem_pad = SMEM_PAD) {
  int D_SMEM = D + smem_pad;
  return (size_t)BLOCK_N_ * D_SMEM * elt_sz +
         (size_t)PIPE_STAGES_ * BLOCK_K_ * D_SMEM * elt_sz +
         (size_t)PIPE_STAGES_ * BLOCK_K_ * sizeof(float);
}

template <typename T, int BLOCK_N_, int BLOCK_K_, int WARPS_, int STAGES_,
          int N_TILES_ = 1, bool ASYNC_CSQ_ = false, int D_FIXED_ = 0,
          bool RAW_DIST_ = false, bool FP32_ACC_ = false,
          bool SIMILARITY_ = false, int SMEM_PAD_ = SMEM_PAD>
inline void launch_typed(
    const at::Tensor& x, const at::Tensor& centroids,
    const at::Tensor& x_sq, const at::Tensor& c_sq,
    at::Tensor& cluster_ids,
    int B, int N, int K, int D, cudaStream_t stream) {
  const int d_tile = (D_FIXED_ > 0) ? D_FIXED_ : D;
  size_t smem_bytes = compute_smem_bytes(BLOCK_N_, BLOCK_K_, d_tile, STAGES_, sizeof(T), SMEM_PAD_);
  auto fn = assign_sm80_kernel<T, BLOCK_N_, BLOCK_K_, WARPS_, STAGES_,
                               N_TILES_, ASYNC_CSQ_, D_FIXED_, RAW_DIST_,
                               FP32_ACC_, SIMILARITY_, SMEM_PAD_>;
  if (smem_bytes > 48 * 1024) {
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_bytes));
  }
  const int rows_per_cta = BLOCK_N_ * N_TILES_;
  dim3 grid((N + rows_per_cta - 1) / rows_per_cta, B);
  dim3 block(WARPS_ * 32);
  fn<<<grid, block, smem_bytes, stream>>>(
      reinterpret_cast<const T*>(x.data_ptr()),
      reinterpret_cast<const T*>(centroids.data_ptr()),
      SIMILARITY_ ? nullptr : x_sq.data_ptr<float>(),
      SIMILARITY_ ? nullptr : c_sq.data_ptr<float>(),
      cluster_ids.data_ptr<int32_t>(),
      B, N, K, D);
}

// NOTE: RAW_DIST_ defaults to true here (not false) to preserve the behavior
// of existing callers in assign_sm80.cu that do not pass RAW_DIST_ explicitly.
// There are 15 call sites with fewer than 8 explicit template parameters that
// rely on this default. Changing it to false would silently alter their
// behavior. The task description's note about changing to false was predicated
// on "every existing caller passes RAW_DIST_ explicitly" — which is not the
// case. See concern reported in commit message.
template <typename T, int BLOCK_N_, int BLOCK_K_, int WARPS_, int STAGES_,
          int N_TILES_ = 1, int D_FIXED_ = 0, bool RAW_DIST_ = true,
          bool FP32_ACC_ = false, int SMEM_PAD_ = SMEM_PAD>
inline void launch_typed_select_csq(
    const at::Tensor& x, const at::Tensor& centroids,
    const at::Tensor& x_sq, const at::Tensor& c_sq,
    at::Tensor& cluster_ids,
    int B, int N, int K, int D, cudaStream_t stream, bool async_csq,
    bool similarity = false) {
  if (similarity) {
    launch_typed<T, BLOCK_N_, BLOCK_K_, WARPS_, STAGES_, N_TILES_, false, D_FIXED_, RAW_DIST_, FP32_ACC_, true, SMEM_PAD_>(
        x, centroids, x_sq, c_sq, cluster_ids, B, N, K, D, stream);
  } else if (async_csq) {
    launch_typed<T, BLOCK_N_, BLOCK_K_, WARPS_, STAGES_, N_TILES_, true, D_FIXED_, RAW_DIST_, FP32_ACC_, false, SMEM_PAD_>(
        x, centroids, x_sq, c_sq, cluster_ids, B, N, K, D, stream);
  } else {
    launch_typed<T, BLOCK_N_, BLOCK_K_, WARPS_, STAGES_, N_TILES_, false, D_FIXED_, RAW_DIST_, FP32_ACC_, false, SMEM_PAD_>(
        x, centroids, x_sq, c_sq, cluster_ids, B, N, K, D, stream);
  }
}

}  // namespace assign
}  // namespace fkc
