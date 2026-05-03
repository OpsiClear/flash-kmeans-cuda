// flash_kmeans_cuda/csrc/assign/assign_autotune.cu
//
// In-memory autotuner for the assign dispatcher. Probes the top 3
// SMEM-feasible candidates per cold (dtype, D, k_bucket) cell, picks the
// fastest by min-of-3 cudaEvent timing, caches the result. Hot path is
// lock-free.

#include "assign_autotune.h"

#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>
#include <limits>
#include <utility>

namespace fkc {
namespace assign {

namespace {

constexpr int kProbeTopN  = 3;
constexpr int kProbeIters = 3;

float time_one_launch(LaunchCtx& ctx, const Variant* v, bool is_fp16,
                      cudaEvent_t start, cudaEvent_t stop) {
  cudaEventRecord(start, ctx.stream);
  bool ok = is_fp16 ? v->try_fp16(ctx) : v->try_bf16(ctx);
  cudaEventRecord(stop, ctx.stream);
  cudaEventSynchronize(stop);
  if (!ok) return std::numeric_limits<float>::infinity();
  float ms = 0.f;
  cudaEventElapsedTime(&ms, start, stop);
  return ms;
}

}  // namespace

VariantView
AutotuneCache::get_or_probe(AutotuneKey key,
                             LaunchCtx& ctx,
                             VariantView static_candidates,
                             bool autotune_enabled,
                             bool verbose) {
  if (!autotune_enabled) return static_candidates;

  Cell& cell = cells_[key.dtype_idx][key.d_idx][key.k_bucket];
  if (cell.probed.load(std::memory_order_acquire)) {
    return VariantView(cell.ordered.data(), MAX_CAND);
  }

  std::lock_guard<std::mutex> g(probe_mu_);
  if (cell.probed.load(std::memory_order_relaxed)) {
    return VariantView(cell.ordered.data(), MAX_CAND);
  }

  // Filter SMEM-feasible candidates.
  std::array<const Variant*, MAX_CAND> feasible{};
  int n_feasible = 0;
  for (size_t i = 0; i < static_candidates.size(); ++i) {
    const Variant* v = static_candidates[i];
    if (!v) break;
    if (v->smem(ctx.D, ctx.elt_sz) <= ctx.smem_limit) {
      feasible[n_feasible++] = v;
    }
  }
  if (n_feasible == 0) {
    // Nothing fits — leave cell empty so the dispatcher falls through to the
    // safe kernel.
    cell.probed.store(true, std::memory_order_release);
    return VariantView(cell.ordered.data(), MAX_CAND);
  }

  // Probe the top-N feasible.
  bool is_fp16 = (key.dtype_idx == 0);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int n_probe = std::min(n_feasible, kProbeTopN);
  std::array<std::pair<float, const Variant*>, kProbeTopN> timings{};
  for (int i = 0; i < n_probe; ++i) {
    const Variant* v = feasible[i];
    // Warmup.
    time_one_launch(ctx, v, is_fp16, start, stop);
    float best = std::numeric_limits<float>::infinity();
    for (int j = 0; j < kProbeIters; ++j) {
      float t = time_one_launch(ctx, v, is_fp16, start, stop);
      if (t < best) best = t;
    }
    timings[i] = {best, v};
    if (verbose) {
      std::fprintf(stderr,
        "[fkc autotune] dtype=%s D_idx=%d k_bucket=%d probe[%d]=%s -> %.3f ms\n",
        is_fp16 ? "fp16" : "bf16", key.d_idx, key.k_bucket,
        i, v->name, best);
    }
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Sort probed by time, then append remaining feasible (un-probed) as
  // fallback.
  std::sort(timings.begin(), timings.begin() + n_probe,
            [](const auto& a, const auto& b) { return a.first < b.first; });
  int out = 0;
  for (int i = 0; i < n_probe; ++i) cell.ordered[out++] = timings[i].second;
  for (int i = kProbeTopN; i < n_feasible && out < MAX_CAND; ++i) {
    cell.ordered[out++] = feasible[i];
  }
  while (out < MAX_CAND) cell.ordered[out++] = nullptr;

  cell.probed.store(true, std::memory_order_release);
  return VariantView(cell.ordered.data(), MAX_CAND);
}

AutotuneCache& autotune_cache() {
  static AutotuneCache c;
  return c;
}

}  // namespace assign
}  // namespace fkc
