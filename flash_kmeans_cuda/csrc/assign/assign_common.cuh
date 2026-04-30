#pragma once

// Shared helpers for the assign kernels (tile constants, in-register reduction,
// per-point min update). Pulled out so the sm80/sm90/fp32 kernels can share
// the epilogue logic.

#include "../common/arch.cuh"
#include <cfloat>
#include <cstdint>

namespace fkc {
namespace assign {

// Per-point running best held in registers.
struct Best {
  float dist;
  int   idx;
};

// Update best given a candidate (d, k). Lower distance wins; on ties keep the
// existing (smaller) index — matches how a streaming min-reduction over k
// preserves the first occurrence.
__device__ __forceinline__ void update_best(Best& b, float d, int k) {
  if (d < b.dist) {
    b.dist = d;
    b.idx = k;
  }
}

// Convert raw cross-product into squared Euclidean distance.
//   dist = x_sq + c_sq - 2 * cross
// Clamped to >= 0 to match the Triton reference (assign_euclid_triton.py:507).
__device__ __forceinline__ float to_dist(float cross, float x_sq, float c_sq) {
  float d = x_sq + c_sq - 2.0f * cross;
  return d > 0.0f ? d : 0.0f;
}

}  // namespace assign
}  // namespace fkc
