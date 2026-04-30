#pragma once

// Architecture probes and dtype traits shared across kernels.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace fkc {

// Compile-time arch guards.
#if defined(__CUDA_ARCH__)
  #define FKC_CUDA_ARCH __CUDA_ARCH__
#else
  #define FKC_CUDA_ARCH 0
#endif

#define FKC_HAS_MMA_SYNC_FP16   (FKC_CUDA_ARCH >= 800)   // m16n8k16 fp16
#define FKC_HAS_CP_ASYNC        (FKC_CUDA_ARCH >= 800)
#define FKC_HAS_LDMATRIX_X4     (FKC_CUDA_ARCH >= 750)
#define FKC_HAS_WGMMA           (FKC_CUDA_ARCH == 900)   // sm_90a only — Phase B

// dtype traits.
template <typename T>
struct dtype_traits;

template <>
struct dtype_traits<__half> {
  using packed2 = __half2;
  static constexpr int torch_scalar_type = 5;  // torch::kHalf
  static constexpr const char* name = "fp16";
  static constexpr bool is_half = true;
};

template <>
struct dtype_traits<__nv_bfloat16> {
  using packed2 = __nv_bfloat162;
  static constexpr int torch_scalar_type = 15; // torch::kBFloat16
  static constexpr const char* name = "bf16";
  static constexpr bool is_half = true;
};

template <>
struct dtype_traits<float> {
  using packed2 = float2;
  static constexpr int torch_scalar_type = 6;  // torch::kFloat
  static constexpr const char* name = "fp32";
  static constexpr bool is_half = false;
};

// Useful constants.
constexpr int kWarp = 32;

}  // namespace fkc
