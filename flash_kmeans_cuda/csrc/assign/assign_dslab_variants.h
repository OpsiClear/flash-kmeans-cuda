// flash_kmeans_cuda/csrc/assign/assign_dslab_variants.h
#pragma once

// D-slab variant factory: per-D template instantiation that holds one
// 128-wide D-slab in SMEM at a time. SMEM footprint is D-independent at
// SLAB_MAX=128, so every target D gets the same BN=128 BK=128 STAGES=2
// 8-warp tile that the legacy kernel uses successfully at D=128.

#include "assign_kernel_launch.h"
#include "assign_variants.h"

#include <ATen/core/Tensor.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <array>
#include <cstddef>

namespace fkc {
namespace assign {

constexpr int SLAB_MAX = 128;
constexpr int MAX_SLABS = 4;

// Forward decl of the kernel template defined in assign_sm80_dslab_kernel.cuh.
//
// IMPORTANT: signature, __launch_bounds__, and __restrict__ qualifiers must
// stay in sync with the kernel definition. NVCC bakes __launch_bounds__ into
// the kernel symbol; mismatch causes cudaErrorInvalidDeviceFunction at runtime.
template <typename T,
          int BLOCK_N, int BLOCK_K, int WARPS_PER_CTA, int PIPE_STAGES,
          int N_TILES_PER_CTA,
          int D_FULL,
          int S0, int S1, int S2, int S3,
          int SMEM_PAD_SLAB,
          bool RAW_DIST>
__global__ void __launch_bounds__(WARPS_PER_CTA * 32, 1)
assign_sm80_dslab_kernel(const T* __restrict__, const T* __restrict__,
                         const float* __restrict__, const float* __restrict__,
                         int32_t* __restrict__,
                         int, int, int, int);

template <int BN, int BK, int W, int S, int NT,
          int D_FULL,
          int S0, int S1, int S2, int S3,
          int SMEM_PAD_SLAB = 0,
          bool Raw = true>
struct DSlabVariantSpec {
  static size_t smem(int /*D*/, size_t elt) {
    // The kernel scaffold (Tasks 2-4) still uses D_FULL as its working
    // dimension; Task 5 will restructure to one SLAB_MAX-wide slab at a
    // time. While in scaffold mode, we must allocate enough SMEM for the
    // legacy-style full-D layout, otherwise the kernel walks off the end
    // of its allocation. Once Task 5 lands, swap D_FULL for SLAB_MAX here
    // to realize the D-independent footprint.
    size_t row = (size_t)(D_FULL + SMEM_PAD_SLAB);
    return BN * row * elt
         + (size_t)S * BK * row * elt
         + (size_t)S * BK * sizeof(float);
  }

  template <class T>
  static bool try_launch(LaunchCtx& c) {
    if (c.D != D_FULL) return false;
    if (smem(c.D, c.elt_sz) > c.smem_limit) return false;
    cudaStream_t stream = c.stream;
    size_t smem_bytes = smem(c.D, c.elt_sz);
    auto fn = assign_sm80_dslab_kernel<
        T, BN, BK, W, S, NT, D_FULL, S0, S1, S2, S3, SMEM_PAD_SLAB, Raw>;
    if (smem_bytes > 48 * 1024) {
      cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(smem_bytes));
    }
    const int rows_per_cta = BN * NT;
    dim3 grid((c.N + rows_per_cta - 1) / rows_per_cta, c.B);
    dim3 block(W * 32);
    fn<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const T*>(c.x.data_ptr()),
        reinterpret_cast<const T*>(c.centroids.data_ptr()),
        c.x_sq.data_ptr<float>(), c.c_sq.data_ptr<float>(),
        c.cluster_ids.data_ptr<int32_t>(),
        c.B, c.N, c.K, c.D);
    return true;
  }
};

template <int BN, int BK, int W, int S, int NT,
          int D_FULL,
          int S0, int S1, int S2, int S3,
          int SMEM_PAD_SLAB = 0,
          bool Raw = true>
constexpr Variant make_dslab_variant(const char* name) {
  using Spec = DSlabVariantSpec<BN, BK, W, S, NT, D_FULL,
                                S0, S1, S2, S3, SMEM_PAD_SLAB, Raw>;
  return Variant{
    name,
    &Spec::template try_launch<__half>,
    &Spec::template try_launch<__nv_bfloat16>,
    &Spec::smem,
  };
}

}  // namespace assign
}  // namespace fkc
