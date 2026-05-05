// flash_kmeans_cuda/csrc/assign/assign_variants.h
#pragma once

// Variant factory: each kernel shape (BN, BK, WARPS, STAGES, N_TILES, D_FIXED,
// RAW, KMIN) is wrapped in a VariantSpec template. make_variant<…> produces a
// constexpr Variant struct holding name + dtype-erased function pointers, so
// the policy table and autotuner can hold catalog entries by const Variant*.

#include "assign_kernel_launch.h"

#include <ATen/core/Tensor.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstddef>

namespace fkc {
namespace assign {

struct LaunchCtx {
  const at::Tensor& x;
  const at::Tensor& centroids;
  const at::Tensor& x_sq;
  const at::Tensor& c_sq;
  at::Tensor& cluster_ids;
  int B, N, K, D;
  size_t elt_sz;
  size_t smem_limit;
  bool async_csq;
  cudaStream_t stream;
};

template <int BN, int BK, int W, int S, int NT, int DFix = 0, bool Raw = false,
          int KMin = 0, int Pad = SMEM_PAD>
struct VariantSpec {
  static size_t smem(int D, size_t elt) {
    return compute_smem_bytes(BN, BK, D, S, elt, Pad);
  }

  template <class T>
  static bool try_launch(LaunchCtx& c) {
    if constexpr (DFix != 0) {
      if (c.D != DFix) return false;
    }
    if constexpr (KMin > 0) {
      if (c.K < KMin) return false;
    }
    if (smem(c.D, c.elt_sz) > c.smem_limit) return false;
    launch_typed_select_csq<T, BN, BK, W, S, NT, DFix, Raw, Pad>(
        c.x, c.centroids, c.x_sq, c.c_sq, c.cluster_ids,
        c.B, c.N, c.K, c.D, c.stream, c.async_csq);
    return true;
  }
};

struct Variant {
  const char* name;
  bool   (*try_fp16)(LaunchCtx&);
  bool   (*try_bf16)(LaunchCtx&);
  size_t (*smem)(int D, size_t elt);
};

template <int BN, int BK, int W, int S, int NT, int DFix = 0, bool Raw = false,
          int KMin = 0, int Pad = SMEM_PAD>
constexpr Variant make_variant(const char* name) {
  using Spec = VariantSpec<BN, BK, W, S, NT, DFix, Raw, KMin, Pad>;
  return Variant{
    name,
    &Spec::template try_launch<__half>,
    &Spec::template try_launch<__nv_bfloat16>,
    &Spec::smem,
  };
}

}  // namespace assign
}  // namespace fkc
