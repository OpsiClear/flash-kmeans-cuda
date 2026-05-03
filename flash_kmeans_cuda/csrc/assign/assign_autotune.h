// flash_kmeans_cuda/csrc/assign/assign_autotune.h
#pragma once

#include "assign_policy.h"

#include <array>
#include <atomic>
#include <mutex>

namespace fkc {
namespace assign {

struct AutotuneKey {
  int dtype_idx;
  int d_idx;
  int k_bucket;
};

class AutotuneCache {
 public:
  // Returns the ordered candidate list for this (dtype, D, k_bucket) cell.
  // On first miss, runs a probe over the top-3 SMEM-feasible candidates from
  // `static_candidates` and caches the winner ordering. Subsequent calls are
  // lock-free.
  //
  // If autotune_enabled is false, returns static_candidates unchanged.
  VariantView get_or_probe(
      AutotuneKey key,
      LaunchCtx& ctx,
      VariantView static_candidates,
      bool autotune_enabled,
      bool verbose);

 private:
  struct Cell {
    std::array<const Variant*, MAX_CAND> ordered{};
    std::atomic<bool> probed{false};
  };
  // [dtype][d_idx][k_bucket] — total 2 * 9 * 5 = 90 cells.
  std::array<std::array<std::array<Cell, N_K_BUCKET>, N_D_IDX>, N_DTYPES> cells_;
  std::mutex probe_mu_;
};

AutotuneCache& autotune_cache();

}  // namespace assign
}  // namespace fkc
