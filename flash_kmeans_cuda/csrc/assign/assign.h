#pragma once

// Host-side dispatch surface for the assignment kernels.
//
// Two paths exist in Phase A:
//   launch_assign_safe — works for fp16/bf16/fp32, no tensor cores. Correctness
//     baseline; default for all dtypes.
//   launch_assign_sm80 — hand-rolled m16n8k16 mma.sync, fp16/bf16. Marked
//     EXPERIMENTAL: needs hardware validation against assign_safe before being
//     trusted. Opt in via env var FKC_ASSIGN_USE_MMA=1.

// Kernel TUs only need at::Tensor. We use <ATen/ATen.h> rather than
// <torch/torch.h> or <torch/extension.h> to avoid pulling Python.h (which
// brings Windows.h's `#define small char`) and to avoid
// torch/csrc/dynamo/compiled_autograd.h (which has an std-namespace
// ambiguity bug with CUDA 13 / CCCL / /Zc:preprocessor on Windows).
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

namespace fkc {
namespace assign {

void launch_assign_safe(const at::Tensor& x,
                        const at::Tensor& centroids,
                        const at::Tensor& x_sq,
                        const at::Tensor& c_sq,
                        at::Tensor& cluster_ids);

// EXPERIMENTAL: not yet validated on hardware. See header comment in
// assign_sm80.cu for the operand layout assumptions that need verification.
void launch_assign_sm80(const at::Tensor& x,
                        const at::Tensor& centroids,
                        const at::Tensor& x_sq,
                        const at::Tensor& c_sq,
                        at::Tensor& cluster_ids);

}  // namespace assign
}  // namespace fkc
