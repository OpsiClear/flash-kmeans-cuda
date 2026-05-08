// flash_kmeans_cuda/csrc/assign/assign_policy.h
#pragma once

#include "assign_variants.h"

#include <array>
#include <cstddef>

// Note: std::span requires C++20; this project builds with -std=c++17.
// ArrayView<T> is a minimal non-owning view substitute used in place of
// std::span<const Variant* const> throughout this header and assign_policy.cu.

namespace fkc {
namespace assign {

// === Minimal non-owning view (std::span substitute for C++17) ================

template <typename T>
struct ArrayView {
  T*     data_;
  size_t size_;

  constexpr ArrayView(T* data, size_t size) noexcept : data_(data), size_(size) {}

  constexpr T*     data()  const noexcept { return data_; }
  constexpr size_t size()  const noexcept { return size_; }
  constexpr T*     begin() const noexcept { return data_; }
  constexpr T*     end()   const noexcept { return data_ + size_; }
  constexpr T&     operator[](size_t i) const noexcept { return data_[i]; }
};

// Convenience alias matching the original std::span usage in the task spec.
using VariantView = ArrayView<const Variant* const>;

// === Index/bucket helpers ====================================================

constexpr int N_DTYPES   = 2;   // 0=fp16, 1=bf16
constexpr int N_D_IDX    = 25;  // {1..16,64,96,128,192,224,256,320,384,OTHER}
constexpr int N_K_BUCKET = 5;   // tiny, small, med, large, mega
constexpr int MAX_CAND   = 8;   // upper bound of candidates per cell

constexpr int OTHER_D_IDX = 24;

inline int dtype_index_of(at::ScalarType t) {
  if (t == at::kHalf)      return 0;
  if (t == at::kBFloat16)  return 1;
  return -1;
}

constexpr int d_index_of(int D) {
  switch (D) {
    case   1: return 0;
    case   2: return 1;
    case   3: return 2;
    case   4: return 3;
    case   5: return 4;
    case   6: return 5;
    case   7: return 6;
    case   8: return 7;
    case   9: return 8;
    case  10: return 9;
    case  11: return 10;
    case  12: return 11;
    case  13: return 12;
    case  14: return 13;
    case  15: return 14;
    case  16: return 15;
    case  64: return 16;
    case  96: return 17;
    case 128: return 18;
    case 192: return 19;
    case 224: return 20;
    case 256: return 21;
    case 320: return 22;
    case 384: return 23;
    default:  return OTHER_D_IDX;
  }
}

constexpr int k_bucket_of(int K) {
  if (K <  128) return 0;
  if (K <  512) return 1;
  if (K < 2048) return 2;
  if (K < 8192) return 3;
  return 4;
}

// === Policy table accessor ===================================================

// Null-terminated row of candidate Variants; cells with fewer than MAX_CAND
// candidates have the trailing slots set to nullptr.
using PolicyRow = std::array<const Variant*, MAX_CAND>;

VariantView static_policy(int dtype_idx, int d_idx, int k_bucket, int n_tiles_override = 0);

// === Per-call env knobs ======================================================

struct EnvKnobs {
  int  n_tiles_override = 0;     // FKC_NTILES; 0 = unset
  bool wide3            = false; // FKC_WIDE3
  bool w4               = false; // FKC_W4
  bool narrow           = false; // FKC_NARROW
  bool deep             = false; // FKC_ASSIGN_DEEP_TILE
  bool dslab            = false; // FKC_DSLAB — force-route to D-slab variant
  bool autotune         = true;  // FKC_AUTOTUNE (default on)
  bool verbose          = false; // FKC_AUTOTUNE_VERBOSE

  bool has_force_override() const {
    return wide3 || w4 || narrow || deep || dslab || n_tiles_override != 0;
  }
};

EnvKnobs read_env_knobs();

// Build the candidate list when the user has forced a specific shape via
// FKC_* env vars. Mirrors the existing force_*_env branches in assign_sm80.cu.
VariantView build_forced_candidates(const EnvKnobs& knobs, const LaunchCtx& ctx);

}  // namespace assign
}  // namespace fkc
