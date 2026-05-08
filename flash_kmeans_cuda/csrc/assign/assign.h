#pragma once

// Host-side dispatch surface for the assignment kernels.
//
// Two paths exist:
//   launch_assign_safe — works for fp16/bf16/fp32 without tensor cores. It is
//     the fp32 path and the final fallback for unsupported or unfitted shapes.
//   launch_assign_sm80 / launch_similarity_assign_sm80 — hand-rolled
//     m16n8k16 mma.sync dispatchers for fp16/bf16 on SM80+ GPUs. Euclidean
//     assignment owns the variant policy table and autotuner; similarity uses
//     the same policy rows with an argmax-dot epilogue.

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

void launch_similarity_assign_safe(const at::Tensor& x,
                                   const at::Tensor& centroids,
                                   at::Tensor& cluster_ids);

void launch_assign_sm80(const at::Tensor& x,
                        const at::Tensor& centroids,
                        const at::Tensor& x_sq,
                        const at::Tensor& c_sq,
                        at::Tensor& cluster_ids);

void launch_similarity_assign_sm80(const at::Tensor& x,
                                   const at::Tensor& centroids,
                                   at::Tensor& cluster_ids);

}  // namespace assign
}  // namespace fkc
