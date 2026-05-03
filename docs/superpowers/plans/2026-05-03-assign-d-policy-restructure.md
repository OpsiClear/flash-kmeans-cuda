# Assign D-by-D Policy Restructure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the tangled `if/else` dispatcher in `launch_assign_sm80` with a factory-based variant catalog, a per-`(dtype, D, K_bucket)` static policy table, and an in-memory autotuner that probes the top-3 candidates per cold cell. Adds full wide-tile + raw-distance specialization for D ∈ {64, 96, 128, 192, 224, 256, 320, 384}.

**Architecture:** Three new files (`assign_variants.h`, `assign_policy.cu`, `assign_autotune.h`/`.cu`) wrap the existing `launch_typed_select_csq<…>` template instantiations. `launch_assign_sm80` shrinks to ~30 lines: build `LaunchCtx`, look up cached candidate list, loop calling `try_fp16` / `try_bf16`. Env knobs become per-call (closes the static-at-first-launch gap).

**Tech Stack:** CUDA / nvcc, ATen / libtorch, nanobind extension, pytest, `setup.py` (CUDAExtension).

**Source spec:** `docs/superpowers/specs/2026-05-03-assign-d-policy-restructure-design.md` (commit `2f2ae4c`).

---

## File Map

| Path                                                         | Action   | Responsibility                                                                                                                             |
|--------------------------------------------------------------|----------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `flash_kmeans_cuda/csrc/assign/assign_variants.h`            | Create   | Header-only. `LaunchCtx`, `VariantSpec<…>`, `Variant`, `make_variant<…>`. No translation-unit state.                                       |
| `flash_kmeans_cuda/csrc/assign/assign_policy.h`              | Create   | Declarations: `AutotuneKey`, `k_bucket_of`, `d_index_of`, `dtype_index_of`, `static_policy(dtype_idx, d_idx, k_bucket)`, `EnvKnobs`, `read_env_knobs`. |
| `flash_kmeans_cuda/csrc/assign/assign_policy.cu`             | Create   | All `constexpr Variant V_*` definitions + `kStaticPolicy` table + `EnvKnobs` reader + helper functions.                                    |
| `flash_kmeans_cuda/csrc/assign/assign_autotune.h`            | Create   | `AutotuneCache` class declaration + `autotune_cache()` accessor.                                                                           |
| `flash_kmeans_cuda/csrc/assign/assign_autotune.cu`           | Create   | `AutotuneCache` definition; probe loop with `cudaEvent_t` timing.                                                                          |
| `flash_kmeans_cuda/csrc/assign/assign_sm80.cu`               | Modify   | Move template-launch helpers (`launch_typed`, `launch_typed_select_csq`, `compute_smem_bytes`) to a header so the catalog can reach them. Replace `launch_assign_sm80` body with the new dispatch loop. |
| `flash_kmeans_cuda/csrc/assign/assign_kernel_launch.h`       | Create   | Move `compute_smem_bytes`, `launch_typed`, `launch_typed_select_csq` here so `assign_policy.cu` can include them.                          |
| `setup.py`                                                   | Modify   | Add `assign_policy.cu` and `assign_autotune.cu` to the source list.                                                                        |
| `tests/test_assign_dispatch_equiv.py`                        | Create   | Stage 1 numerical-equivalence test + Stage 2 D-coverage test.                                                                              |
| `tests/test_assign_autotune.py`                              | Create   | Stage 3 autotune-cache, verbose-log, and thread-safety tests.                                                                              |
| `tests/test_persistent.py`                                   | Modify   | Replace subprocess-per-FKC_NTILES boilerplate with in-process toggling once env knobs are per-call.                                        |

Stages map to commits: each task ends with a commit so partial progress is checkpoint-able.

---

## Stage 1 — Refactor (no behavior change)

### Task 1: Extract template launchers into a reusable header

**Files:**
- Create: `flash_kmeans_cuda/csrc/assign/assign_kernel_launch.h`
- Modify: `flash_kmeans_cuda/csrc/assign/assign_sm80.cu:646-704` (delete the moved code, leave a `#include`)

- [ ] **Step 1: Create the header**

```cpp
// flash_kmeans_cuda/csrc/assign/assign_kernel_launch.h
#pragma once

// Template-launch helpers for the sm80 assign kernel. Lives in a header so
// the variant catalog (assign_policy.cu) can reach them without forcing every
// dispatch entry through assign_sm80.cu's translation unit.

#include "assign.h"
#include "assign_common.cuh"
#include "../common/arch.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace fkc {
namespace assign {

// Forward decl of the kernel template defined in assign_sm80.cu.
template <typename T, int BLOCK_N_, int BLOCK_K_, int WARPS_, int STAGES_,
          int N_TILES_, bool ASYNC_CSQ_, int D_FIXED_, bool RAW_DIST_>
__global__ void assign_sm80_kernel(const T*, const T*, const float*, const float*,
                                   int32_t*, int, int, int, int);

inline size_t compute_smem_bytes(int BLOCK_N_, int BLOCK_K_, int D,
                                 int PIPE_STAGES_, size_t elt_sz) {
  int D_SMEM = D + SMEM_PAD;
  return (size_t)BLOCK_N_ * D_SMEM * elt_sz +
         (size_t)PIPE_STAGES_ * BLOCK_K_ * D_SMEM * elt_sz +
         (size_t)PIPE_STAGES_ * BLOCK_K_ * sizeof(float);
}

template <typename T, int BLOCK_N_, int BLOCK_K_, int WARPS_, int STAGES_,
          int N_TILES_ = 1, bool ASYNC_CSQ_ = false, int D_FIXED_ = 0,
          bool RAW_DIST_ = false>
inline void launch_typed(
    const at::Tensor& x, const at::Tensor& centroids,
    const at::Tensor& x_sq, const at::Tensor& c_sq,
    at::Tensor& cluster_ids,
    int B, int N, int K, int D, cudaStream_t stream) {
  size_t smem_bytes = compute_smem_bytes(BLOCK_N_, BLOCK_K_, D, STAGES_, sizeof(T));
  auto fn = assign_sm80_kernel<T, BLOCK_N_, BLOCK_K_, WARPS_, STAGES_,
                               N_TILES_, ASYNC_CSQ_, D_FIXED_, RAW_DIST_>;
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
      x_sq.data_ptr<float>(), c_sq.data_ptr<float>(),
      cluster_ids.data_ptr<int32_t>(),
      B, N, K, D);
}

template <typename T, int BLOCK_N_, int BLOCK_K_, int WARPS_, int STAGES_,
          int N_TILES_ = 1, int D_FIXED_ = 0, bool RAW_DIST_ = false>
inline void launch_typed_select_csq(
    const at::Tensor& x, const at::Tensor& centroids,
    const at::Tensor& x_sq, const at::Tensor& c_sq,
    at::Tensor& cluster_ids,
    int B, int N, int K, int D, cudaStream_t stream, bool async_csq) {
  if (async_csq) {
    launch_typed<T, BLOCK_N_, BLOCK_K_, WARPS_, STAGES_, N_TILES_, true, D_FIXED_, RAW_DIST_>(
        x, centroids, x_sq, c_sq, cluster_ids, B, N, K, D, stream);
  } else {
    launch_typed<T, BLOCK_N_, BLOCK_K_, WARPS_, STAGES_, N_TILES_, false, D_FIXED_, RAW_DIST_>(
        x, centroids, x_sq, c_sq, cluster_ids, B, N, K, D, stream);
  }
}

}  // namespace assign
}  // namespace fkc
```

Note: `launch_typed_select_csq`'s old default `RAW_DIST_ = true` is fixed to `false` here to match the actual template default in `assign_sm80.cu` (which is `bool RAW_DIST_ = false` on `launch_typed`). The original `= true` in `launch_typed_select_csq` was harmless because every caller explicitly supplied the value, but the new factory will rely on the default.

- [ ] **Step 2: Delete the moved code from `assign_sm80.cu` and add the include**

In `flash_kmeans_cuda/csrc/assign/assign_sm80.cu`, delete lines 646–704 (the `compute_smem_bytes`, `launch_typed`, and `launch_typed_select_csq` definitions) and add near the top:

```cpp
#include "assign_kernel_launch.h"
```

- [ ] **Step 3: Build and run existing tests to confirm no behavior change**

Run: `uv run python setup.py build_ext --inplace`
Expected: clean build.

Run: `uv run pytest tests/test_correctness.py tests/test_persistent.py tests/test_shapes.py -x -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add flash_kmeans_cuda/csrc/assign/assign_kernel_launch.h flash_kmeans_cuda/csrc/assign/assign_sm80.cu
git commit -m "Extract assign template launchers into reusable header"
```

---

### Task 2: Add the variant factory header

**Files:**
- Create: `flash_kmeans_cuda/csrc/assign/assign_variants.h`

- [ ] **Step 1: Write the header**

```cpp
// flash_kmeans_cuda/csrc/assign/assign_variants.h
#pragma once

// Variant factory: each kernel shape (BN, BK, WARPS, STAGES, N_TILES, D_FIXED,
// RAW) is wrapped in a VariantSpec template. make_variant<…> produces a
// constexpr Variant struct holding name + dtype-erased function pointers, so
// the policy table and autotuner can hold catalog entries by const Variant*.

#include "assign_kernel_launch.h"

#include <ATen/ATen.h>
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

template <int BN, int BK, int W, int S, int NT, int DFix = 0, bool Raw = false>
struct VariantSpec {
  static size_t smem(int D, size_t elt) {
    return compute_smem_bytes(BN, BK, D, S, elt);
  }

  template <class T>
  static bool try_launch(LaunchCtx& c) {
    if constexpr (DFix != 0) {
      if (c.D != DFix) return false;
    }
    if (smem(c.D, c.elt_sz) > c.smem_limit) return false;
    launch_typed_select_csq<T, BN, BK, W, S, NT, DFix, Raw>(
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

template <int BN, int BK, int W, int S, int NT, int DFix = 0, bool Raw = false>
constexpr Variant make_variant(const char* name) {
  using Spec = VariantSpec<BN, BK, W, S, NT, DFix, Raw>;
  return Variant{
    name,
    &Spec::template try_launch<__half>,
    &Spec::template try_launch<__nv_bfloat16>,
    &Spec::smem,
  };
}

}  // namespace assign
}  // namespace fkc
```

- [ ] **Step 2: Sanity-build**

Run: `uv run python setup.py build_ext --inplace`
Expected: clean build (header isn't included anywhere yet, but should at minimum parse if pulled in by Task 3).

- [ ] **Step 3: Commit**

```bash
git add flash_kmeans_cuda/csrc/assign/assign_variants.h
git commit -m "Add Variant factory for assign dispatcher"
```

---

### Task 3: Build the variant catalog (existing variants only)

**Files:**
- Create: `flash_kmeans_cuda/csrc/assign/assign_policy.h`
- Create: `flash_kmeans_cuda/csrc/assign/assign_policy.cu`

The catalog at this stage contains *only* the variants already instantiated by today's `try_launch_*` lambdas — no new template instantiations. New ones land in Stage 2.

- [ ] **Step 1: Write `assign_policy.h`**

```cpp
// flash_kmeans_cuda/csrc/assign/assign_policy.h
#pragma once

#include "assign_variants.h"

#include <array>
#include <span>

namespace fkc {
namespace assign {

// === Index/bucket helpers ====================================================

constexpr int N_DTYPES   = 2;   // 0=fp16, 1=bf16
constexpr int N_D_IDX    = 9;   // {64,96,128,192,224,256,320,384,OTHER}
constexpr int N_K_BUCKET = 5;   // tiny, small, med, large, mega
constexpr int MAX_CAND   = 8;   // upper bound of candidates per cell

constexpr int OTHER_D_IDX = 8;

inline int dtype_index_of(at::ScalarType t) {
  if (t == at::kHalf) return 0;
  if (t == at::kBFloat16) return 1;
  return -1;
}

constexpr int d_index_of(int D) {
  switch (D) {
    case  64: return 0;
    case  96: return 1;
    case 128: return 2;
    case 192: return 3;
    case 224: return 4;
    case 256: return 5;
    case 320: return 6;
    case 384: return 7;
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

std::span<const Variant* const>
static_policy(int dtype_idx, int d_idx, int k_bucket);

// === Per-call env knobs ======================================================

struct EnvKnobs {
  int  n_tiles_override = 0;     // FKC_NTILES; 0 = unset
  bool wide3            = false; // FKC_WIDE3
  bool w4               = false; // FKC_W4
  bool narrow           = false; // FKC_NARROW
  bool deep             = false; // FKC_ASSIGN_DEEP_TILE
  bool autotune         = true;  // FKC_AUTOTUNE (default on)
  bool verbose          = false; // FKC_AUTOTUNE_VERBOSE

  bool has_force_override() const {
    return wide3 || w4 || narrow || deep || n_tiles_override != 0;
  }
};

EnvKnobs read_env_knobs();

// Build the candidate list when the user has forced a specific shape via
// FKC_* env vars. Mirrors the existing force_*_env branches in assign_sm80.cu.
std::span<const Variant* const>
build_forced_candidates(const EnvKnobs& knobs, const LaunchCtx& ctx);

}  // namespace assign
}  // namespace fkc
```

- [ ] **Step 2: Write `assign_policy.cu` with catalog covering today's variants only**

```cpp
// flash_kmeans_cuda/csrc/assign/assign_policy.cu
//
// Variant catalog + static policy table. Stage 1 transcribes today's
// dispatch order into table form WITHOUT adding new kernel instantiations.

#include "assign_policy.h"

#include <array>
#include <cstdlib>
#include <cstring>

namespace fkc {
namespace assign {

// =========================================================================
// Variant catalog — every kernel shape currently instantiated.
// =========================================================================

// Wide tiles, BLOCK_N=128.
constexpr Variant V_WIDEK128_W8         = make_variant<128, 128, 8, 2, 1>("widek128_w8");
constexpr Variant V_WIDEK128_W8_D128    = make_variant<128, 128, 8, 2, 1, 128, true>("widek128_w8_d128");
constexpr Variant V_WIDEK128_W8_N2      = make_variant<128, 128, 8, 2, 2>("widek128_w8_n2");
constexpr Variant V_WIDEK128_W8_N2_D64  = make_variant<128, 128, 8, 2, 2,  64, true>("widek128_w8_n2_d64");
constexpr Variant V_WIDEK128_W8_N2_D96  = make_variant<128, 128, 8, 2, 2,  96, true>("widek128_w8_n2_d96");
constexpr Variant V_WIDEK128_W8_N2_D128 = make_variant<128, 128, 8, 2, 2, 128, true>("widek128_w8_n2_d128");
constexpr Variant V_WIDEK128_W8_N4      = make_variant<128, 128, 8, 2, 4>("widek128_w8_n4");
constexpr Variant V_WIDEK128_W8_N4_D128 = make_variant<128, 128, 8, 2, 4, 128, true>("widek128_w8_n4_d128");

constexpr Variant V_WIDEK96_W8          = make_variant<128,  96, 8, 2, 1>("widek96_w8");
constexpr Variant V_WIDEK96_W8_D128     = make_variant<128,  96, 8, 2, 1, 128, true>("widek96_w8_d128");
constexpr Variant V_WIDEK96_W8_N2       = make_variant<128,  96, 8, 2, 2>("widek96_w8_n2");
constexpr Variant V_WIDEK96_W8_N2_D128  = make_variant<128,  96, 8, 2, 2, 128, true>("widek96_w8_n2_d128");
constexpr Variant V_WIDEK96_W8_N4       = make_variant<128,  96, 8, 2, 4>("widek96_w8_n4");
constexpr Variant V_WIDEK96_W8_N4_D128  = make_variant<128,  96, 8, 2, 4, 128, true>("widek96_w8_n4_d128");

// 3-stage wide.
constexpr Variant V_WIDE_3_W8           = make_variant<128,  64, 8, 3, 1>("wide_3_w8");
constexpr Variant V_WIDE_3_W8_N2        = make_variant<128,  64, 8, 3, 2>("wide_3_w8_n2");
constexpr Variant V_WIDE_3_W4           = make_variant<128,  64, 4, 3, 1>("wide_3_w4");
constexpr Variant V_WIDE_2_W4           = make_variant<128,  64, 4, 2, 1>("wide_2_w4");

// Narrow.
constexpr Variant V_NARROW_4            = make_variant< 64,  64, 4, 4, 1>("narrow_4");
constexpr Variant V_NARROWK32_W4        = make_variant< 64,  32, 4, 2, 1>("narrowk32_w4");
constexpr Variant V_NARROWK32_W4_N2     = make_variant< 64,  32, 4, 2, 2>("narrowk32_w4_n2");
constexpr Variant V_NARROWK32_W4_N2_D192= make_variant< 64,  32, 4, 2, 2, 192, true>("narrowk32_w4_n2_d192");
constexpr Variant V_NARROWK32_W4_N2_D224= make_variant< 64,  32, 4, 2, 2, 224, true>("narrowk32_w4_n2_d224");
constexpr Variant V_NARROWK32_W4_N2_D256= make_variant< 64,  32, 4, 2, 2, 256, true>("narrowk32_w4_n2_d256");
constexpr Variant V_NARROWK32_W4_N2_D320= make_variant< 64,  32, 4, 2, 2, 320, true>("narrowk32_w4_n2_d320");
constexpr Variant V_NARROWK32_W4_N2_D384= make_variant< 64,  32, 4, 2, 2, 384, true>("narrowk32_w4_n2_d384");
constexpr Variant V_NARROWK32_W4_N4     = make_variant< 64,  32, 4, 2, 4>("narrowk32_w4_n4");

constexpr Variant V_DEEP_2_W4           = make_variant< 64, 128, 4, 2, 1>("deep_2_w4");

// =========================================================================
// Generic-D fallback chain. Used by every cell as the tail of its candidate
// list, and as the entire row for OTHER_D_IDX.
// =========================================================================
constexpr PolicyRow kGenericFallback = {
  &V_WIDEK128_W8, &V_WIDEK96_W8, &V_WIDE_3_W8,
  &V_WIDE_3_W4, &V_NARROW_4, &V_DEEP_2_W4,
  nullptr, nullptr,
};

// =========================================================================
// Per-D ordered candidate lists (Stage 1: transcribes assign_sm80.cu's
// existing if/else for the n_tiles_choice == 2 default path).
// =========================================================================

constexpr PolicyRow kD64 = {
  &V_WIDEK128_W8_N2_D64, &V_WIDEK128_W8_N2, &V_WIDEK96_W8_N2,
  &V_WIDEK128_W8, &V_WIDEK96_W8, &V_WIDE_3_W8,
  &V_NARROW_4, &V_DEEP_2_W4,
};

constexpr PolicyRow kD96 = {
  &V_WIDEK128_W8_N2_D96, &V_WIDEK128_W8_N2, &V_WIDEK96_W8_N2,
  &V_WIDEK128_W8, &V_WIDEK96_W8, &V_WIDE_3_W8,
  &V_NARROW_4, &V_DEEP_2_W4,
};

constexpr PolicyRow kD128 = {
  &V_WIDEK128_W8_N2_D128, &V_WIDEK128_W8_N2, &V_WIDEK96_W8_N2_D128,
  &V_WIDEK96_W8_N2, &V_WIDEK128_W8_D128, &V_WIDEK96_W8_D128,
  &V_WIDE_3_W8, &V_NARROW_4,
};

constexpr PolicyRow kD192 = {
  &V_NARROWK32_W4_N2_D192, &V_NARROWK32_W4_N2, &V_NARROWK32_W4,
  &V_WIDE_3_W8, &V_WIDE_3_W4, &V_NARROW_4,
  &V_DEEP_2_W4, nullptr,
};

constexpr PolicyRow kD224 = {
  &V_NARROWK32_W4_N2_D224, &V_NARROWK32_W4_N2, &V_NARROWK32_W4,
  &V_WIDE_3_W8, &V_WIDE_3_W4, &V_NARROW_4,
  &V_DEEP_2_W4, nullptr,
};

constexpr PolicyRow kD256 = {
  &V_NARROWK32_W4_N2_D256, &V_NARROWK32_W4_N2, &V_NARROWK32_W4,
  &V_WIDE_3_W8, &V_WIDE_3_W4, &V_NARROW_4,
  &V_DEEP_2_W4, nullptr,
};

constexpr PolicyRow kD320 = {
  &V_NARROWK32_W4_N2_D320, &V_NARROWK32_W4_N2, &V_NARROWK32_W4,
  &V_WIDE_3_W4, &V_NARROW_4, &V_DEEP_2_W4,
  nullptr, nullptr,
};

constexpr PolicyRow kD384 = {
  &V_NARROWK32_W4_N2_D384, &V_NARROWK32_W4_N2, &V_NARROWK32_W4,
  &V_WIDE_3_W4, &V_NARROW_4, &V_DEEP_2_W4,
  nullptr, nullptr,
};

// Per-D rows are duplicated across all K-buckets in Stage 1 (single static
// order, the same as today's if/else, which only switches on K via the
// prefer_w8 boolean — captured implicitly by ordering w8 variants first).
constexpr PolicyRow kRows[N_D_IDX] = {
  kD64, kD96, kD128, kD192, kD224, kD256, kD320, kD384, kGenericFallback,
};

std::span<const Variant* const>
static_policy(int /*dtype_idx*/, int d_idx, int /*k_bucket*/) {
  // Stage 1: identical row regardless of dtype or k_bucket. Stage 3's
  // autotuner reorders within each (dtype,K) cell at runtime.
  return std::span<const Variant* const>(kRows[d_idx].data(), MAX_CAND);
}

// =========================================================================
// Forced-candidate builder for FKC_* env overrides. Mirrors the legacy
// force_*_env branches.
// =========================================================================
namespace {
constexpr PolicyRow kForcedDeep    = { &V_DEEP_2_W4, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
constexpr PolicyRow kForcedNarrowN4= { &V_NARROWK32_W4_N4, &V_NARROWK32_W4, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
constexpr PolicyRow kForcedNarrowN2= { &V_NARROWK32_W4_N2, &V_NARROWK32_W4, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
constexpr PolicyRow kForcedNarrow  = { &V_NARROWK32_W4, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
constexpr PolicyRow kForcedW4      = { &V_WIDE_3_W4, &V_WIDE_2_W4, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
constexpr PolicyRow kForcedWide3N2 = { &V_WIDE_3_W8_N2, &V_WIDE_3_W8, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
constexpr PolicyRow kForcedWide3   = { &V_WIDE_3_W8, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
}  // namespace

std::span<const Variant* const>
build_forced_candidates(const EnvKnobs& knobs, const LaunchCtx& /*ctx*/) {
  // Precedence mirrors today's force_*_env branches: deep > narrow > w4 > wide3.
  if (knobs.deep) {
    return std::span<const Variant* const>(kForcedDeep.data(), MAX_CAND);
  }
  if (knobs.narrow) {
    if (knobs.n_tiles_override == 4) return std::span<const Variant* const>(kForcedNarrowN4.data(), MAX_CAND);
    if (knobs.n_tiles_override >= 2) return std::span<const Variant* const>(kForcedNarrowN2.data(), MAX_CAND);
    return std::span<const Variant* const>(kForcedNarrow.data(), MAX_CAND);
  }
  if (knobs.w4) {
    return std::span<const Variant* const>(kForcedW4.data(), MAX_CAND);
  }
  if (knobs.wide3) {
    if (knobs.n_tiles_override >= 2) return std::span<const Variant* const>(kForcedWide3N2.data(), MAX_CAND);
    return std::span<const Variant* const>(kForcedWide3.data(), MAX_CAND);
  }
  // n_tiles_override on its own keeps the static policy row but pinned: just
  // return the static policy and let the dispatch loop pick the first that fits.
  return static_policy(/*dtype_idx*/0, /*d_idx*/d_index_of(0), /*k_bucket*/0);
}

// =========================================================================
// Env knob reader — called per launch (no static cache).
// =========================================================================
EnvKnobs read_env_knobs() {
  EnvKnobs k;
  if (const char* s = std::getenv("FKC_NTILES")) {
    int v = std::atoi(s);
    if (v == 1 || v == 2 || v == 4) k.n_tiles_override = v;
  }
  auto truthy = [](const char* s) {
    return s && (std::strcmp(s, "1") == 0 || std::strcmp(s, "true") == 0);
  };
  k.wide3    = truthy(std::getenv("FKC_WIDE3"));
  k.w4       = truthy(std::getenv("FKC_W4"));
  k.narrow   = truthy(std::getenv("FKC_NARROW"));
  k.deep     = truthy(std::getenv("FKC_ASSIGN_DEEP_TILE"));
  // FKC_AUTOTUNE defaults to ON. Set FKC_AUTOTUNE=0 to disable.
  if (const char* s = std::getenv("FKC_AUTOTUNE")) {
    k.autotune = !(std::strcmp(s, "0") == 0 || std::strcmp(s, "false") == 0);
  }
  k.verbose  = truthy(std::getenv("FKC_AUTOTUNE_VERBOSE"));
  return k;
}

}  // namespace assign
}  // namespace fkc
```

- [ ] **Step 3: Add `assign_policy.cu` to `setup.py` source list**

Modify `setup.py` (~line 100), adding the new source between `assign_sm80.cu` and `update_sorted.cu`:

```python
sources = [
    str(CSRC / "bindings.cpp"),
    str(CSRC / "assign" / "assign_safe.cu"),
    str(CSRC / "assign" / "assign_sm80.cu"),
    str(CSRC / "assign" / "assign_policy.cu"),
    str(CSRC / "update" / "update_sorted.cu"),
    str(CSRC / "update" / "update_finalize.cu"),
    str(NB_COMBINED),
]
```

- [ ] **Step 4: Build to confirm catalog compiles**

Run: `uv run python setup.py build_ext --inplace`
Expected: clean build. Catalog is not yet referenced by the dispatcher; this only confirms the templates and `make_variant` resolve.

- [ ] **Step 5: Commit**

```bash
git add flash_kmeans_cuda/csrc/assign/assign_policy.h flash_kmeans_cuda/csrc/assign/assign_policy.cu setup.py
git commit -m "Add static variant catalog and policy table for assign dispatcher"
```

---

### Task 4: Replace `launch_assign_sm80` body with the new dispatch loop

**Files:**
- Modify: `flash_kmeans_cuda/csrc/assign/assign_sm80.cu` (the entire `launch_assign_sm80` function body, lines ~707–1154 as of this plan)

- [ ] **Step 1: Rewrite `launch_assign_sm80`**

Replace the entire body of `launch_assign_sm80` (everything from the function signature through the closing `}`) with:

```cpp
void launch_assign_sm80(const at::Tensor& x,
                        const at::Tensor& centroids,
                        const at::Tensor& x_sq,
                        const at::Tensor& c_sq,
                        at::Tensor& cluster_ids) {
  TORCH_CHECK(x.is_cuda() && centroids.is_cuda(),
              "x and centroids must be CUDA tensors");
  TORCH_CHECK(x.dim() == 3 && centroids.dim() == 3,
              "x and centroids must be 3D (B,N,D)/(B,K,D)");
  TORCH_CHECK(x.scalar_type() == centroids.scalar_type(),
              "x and centroids must share dtype");
  TORCH_CHECK(x_sq.scalar_type() == at::kFloat && c_sq.scalar_type() == at::kFloat,
              "x_sq and c_sq must be fp32");
  TORCH_CHECK(cluster_ids.scalar_type() == at::kInt,
              "cluster_ids must be int32");

  int B = x.size(0);
  int N = x.size(1);
  int D = x.size(2);
  int K = centroids.size(1);
  TORCH_CHECK(centroids.size(0) == B && centroids.size(2) == D,
              "centroids must be (B, K, D) matching x");
  TORCH_CHECK(D % BLOCK_D == 0,
              "assign_sm80: D must be a multiple of 16 (got ", D, ")");
  TORCH_CHECK(x.is_contiguous() && centroids.is_contiguous() &&
              x_sq.is_contiguous() && c_sq.is_contiguous() &&
              cluster_ids.is_contiguous(),
              "all tensors must be contiguous");

  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int dev = x.device().index();
  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, dev);
  size_t smem_limit = props.sharedMemPerBlockOptin;
  if (smem_limit == 0) smem_limit = props.sharedMemPerBlock;

  LaunchCtx ctx{
    x, centroids, x_sq, c_sq, cluster_ids,
    B, N, K, D,
    x.element_size(), smem_limit,
    /*async_csq=*/(K >= 256),
    stream,
  };

  EnvKnobs knobs = read_env_knobs();

  std::span<const Variant* const> candidates;
  if (knobs.has_force_override()) {
    candidates = build_forced_candidates(knobs, ctx);
  } else {
    int dtype_idx = dtype_index_of(x.scalar_type());
    TORCH_CHECK(dtype_idx >= 0, "assign_sm80 requires fp16 or bf16 input");
    candidates = static_policy(dtype_idx, d_index_of(D), k_bucket_of(K));
  }

  bool launched = false;
  bool is_fp16 = (x.scalar_type() == at::kHalf);
  for (const Variant* v : candidates) {
    if (!v) break;
    bool ok = is_fp16 ? v->try_fp16(ctx) : v->try_bf16(ctx);
    if (ok) { launched = true; break; }
  }

  if (!launched) {
    launch_assign_safe(x, centroids, x_sq, c_sq, cluster_ids);
    return;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
```

Add the include near the top of `assign_sm80.cu`:

```cpp
#include "assign_policy.h"
```

Note: Stage 3 will swap `static_policy(...)` for `autotune_cache().get_or_probe(...)`. Stage 1 leaves the autotuner stub out so this stage can be tested in isolation.

- [ ] **Step 2: Build**

Run: `uv run python setup.py build_ext --inplace`
Expected: clean build.

- [ ] **Step 3: Run existing tests**

Run: `uv run pytest tests/test_correctness.py tests/test_persistent.py tests/test_shapes.py tests/test_dtypes.py tests/test_mma_optin.py -x -q`
Expected: all pass. (`test_persistent.py` still uses subprocesses since per-call env is just-now enabled; we'll simplify it in Task 6.)

- [ ] **Step 4: Commit**

```bash
git add flash_kmeans_cuda/csrc/assign/assign_sm80.cu
git commit -m "Replace assign dispatcher with policy-table loop"
```

---

### Task 5: Stage 1 numerical-equivalence test

**Files:**
- Create: `tests/test_assign_dispatch_equiv.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_assign_dispatch_equiv.py
"""Stage 1 acceptance: the new policy-table dispatcher must produce the
same cluster_ids as launch_assign_safe for every (D, K) currently exercised
by bench_d_sweep.py. This is a one-off check guarding the Stage 1 refactor;
it's intentionally narrow vs the safe kernel rather than vs Triton (Triton
diverges on tied-distance points; the safe kernel does not)."""

from __future__ import annotations

import pytest
import torch

from flash_kmeans_cuda import _C


def _assign_safe(x, centroids):
    B, N, D = x.shape
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    # Force the safe kernel by passing D not divisible by BLOCK_D? No — the
    # backend dispatch lives in C++. Instead, compute the reference using the
    # numerically deterministic Python path.
    # cluster_ids[b,n] = argmin_k (||x[b,n] - centroids[b,k]||^2).
    diff = x.unsqueeze(2).float() - centroids.unsqueeze(1).float()
    dist = (diff * diff).sum(-1)  # (B, N, K)
    return dist.argmin(dim=-1).to(torch.int32)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "B,N,K,D",
    [
        (1, 4096,  128,  64),
        (1, 4096,  128,  96),
        (1, 4096,  128, 128),
        (1, 4096,  128, 192),
        (1, 4096,  128, 256),
        (1, 4096, 1024, 128),
        (1, 4096, 8192, 128),
    ],
)
def test_dispatcher_matches_python_reference(B, N, K, D, dtype):
    torch.manual_seed(0)
    x = torch.randn(B, N, D, device="cuda", dtype=dtype)
    centroids = x[:, :K].contiguous()
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()

    ids = _C.euclid_assign(x, centroids, x_sq, c_sq, None)
    ref = _assign_safe(x, centroids)

    # mma rounding allows tied-distance differences on a small fraction.
    disagree = (ids != ref).float().mean().item()
    assert disagree < 0.02, (
        f"D={D} K={K} dtype={dtype}: {disagree:.3%} disagreement vs python "
        f"reference (expected <2% — tied-distance rounding only)"
    )
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_assign_dispatch_equiv.py -v`
Expected: all 14 cases pass (7 shapes × 2 dtypes).

- [ ] **Step 3: Commit**

```bash
git add tests/test_assign_dispatch_equiv.py
git commit -m "Add Stage 1 equivalence test for assign dispatcher"
```

---

### Task 6: Simplify `test_persistent.py` to in-process env toggling

Now that env knobs are read per-call, the subprocess-per-FKC_NTILES gymnastics are unnecessary. This isn't required for correctness but proves the per-call read works.

**Files:**
- Modify: `tests/test_persistent.py` (rewrite the three parametric tests; keep `test_ntiles_invariance` on subprocess since it compares two distinct runs)

- [ ] **Step 1: Replace the parametric body with in-process toggling**

Replace the file's `test_identity_centroids`, `test_tail_short_remainder`, and `test_determinism` with:

```python
import os
import pytest
import torch
from flash_kmeans_cuda import _C


def _set_ntiles(ntiles: str):
    os.environ["FKC_NTILES"] = ntiles


@pytest.mark.parametrize("ntiles", ["1", "2", "4"])
def test_identity_centroids(ntiles):
    """First K rows of x as centroids -> argmin trivially picks row index."""
    _set_ntiles(ntiles)
    torch.manual_seed(7)
    B, N, K, D = 1, 4096, 64, 128
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    centroids = x[:, :K].contiguous()
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    ids = _C.euclid_assign(x, centroids, x_sq, c_sq, None)
    expected = torch.arange(K, device="cuda", dtype=torch.int32)
    matches = (ids[0, :K] == expected).float().mean().item()
    assert matches > 0.99, f"FKC_NTILES={ntiles} identity_match={matches:.6f}"


@pytest.mark.parametrize("ntiles", ["1", "2", "4"])
def test_tail_short_remainder(ntiles):
    """N = 257 with BLOCK_N=128 forces a 1-row tail -- shouldn't OOB or break."""
    _set_ntiles(ntiles)
    torch.manual_seed(7)
    B, N, K, D = 1, 257, 16, 128
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    centroids = x[:, :K].contiguous()
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    ids = _C.euclid_assign(x, centroids, x_sq, c_sq, None)
    in_range = ((ids >= 0) & (ids < K)).all().item()
    self_match = (ids[0, :K] == torch.arange(K, device="cuda", dtype=torch.int32)).float().mean().item()
    assert in_range and self_match > 0.95, (
        f"FKC_NTILES={ntiles}: in_range={in_range} self_match={self_match:.4f}"
    )


@pytest.mark.parametrize("ntiles", ["1", "2", "4"])
def test_determinism(ntiles):
    """Same input must yield same cluster_ids on repeated launches."""
    _set_ntiles(ntiles)
    torch.manual_seed(7)
    B, N, K, D = 1, 8192, 128, 128
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    centroids = x[:, :K].contiguous()
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    ids1 = _C.euclid_assign(x, centroids, x_sq, c_sq, None).clone()
    ids2 = _C.euclid_assign(x, centroids, x_sq, c_sq, None).clone()
    assert (ids1 == ids2).all().item(), f"FKC_NTILES={ntiles} non-deterministic"
```

Keep `test_ntiles_invariance` exactly as-is (it intentionally uses subprocesses to compare distinct N_TILES choices).

Update the module docstring:

```python
"""Safety-net tests for the persistent (N_TILES_PER_CTA > 1) kernel path.

Specifically guards against the failure modes Agent 3 flagged:
- cp.async pipeline group counter contamination across n_tile iterations
- best[]/xs_*_cache state leaking between tiles
- Tail handling when N is not a multiple of (BLOCK_N * N_TILES_PER_CTA)

The kernel selects N_TILES via the FKC_NTILES env var, which is read on every
launch, so we toggle it inline within each test.
"""
```

(Delete the old subprocess `_SCRIPT` and `_run` helpers.)

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_persistent.py -v`
Expected: all parametrized cases pass without spawning subprocesses (the test should now run in seconds instead of ~30s).

- [ ] **Step 3: Commit**

```bash
git add tests/test_persistent.py
git commit -m "Toggle FKC_NTILES inline in test_persistent (env now per-call)"
```

---

## Stage 2 — Fill gaps (perf change)

### Task 7: Add new D-specialized variants to the catalog

**Files:**
- Modify: `flash_kmeans_cuda/csrc/assign/assign_policy.cu` (extend the catalog block)

This adds the wide-tile + raw-distance instantiations for D ∈ {64, 96, 192, 224, 256} that don't exist yet, plus the `wide_3` D-specializations for D ∈ {192, 224, 256}. SMEM-infeasible cells (BK=128 at D≥192 etc.) are simply not declared — the templates would still compile, but `Variant::smem` returns >100 KB and `try_launch` rejects at runtime, so leaving them out costs nothing and shrinks the binary.

- [ ] **Step 1: Add the new constexpr Variant lines after the existing catalog**

Append in `assign_policy.cu` after `V_DEEP_2_W4`:

```cpp
// === Stage 2 additions ===================================================
// Wide BK=96 D-specialized (n_tiles=1 and n_tiles=2 each, fp16 + bf16).
constexpr Variant V_WIDEK96_W8_D64       = make_variant<128,  96, 8, 2, 1,  64, true>("widek96_w8_d64");
constexpr Variant V_WIDEK96_W8_D96       = make_variant<128,  96, 8, 2, 1,  96, true>("widek96_w8_d96");
constexpr Variant V_WIDEK96_W8_D192      = make_variant<128,  96, 8, 2, 1, 192, true>("widek96_w8_d192");
constexpr Variant V_WIDEK96_W8_D224      = make_variant<128,  96, 8, 2, 1, 224, true>("widek96_w8_d224");

constexpr Variant V_WIDEK96_W8_N2_D64    = make_variant<128,  96, 8, 2, 2,  64, true>("widek96_w8_n2_d64");
constexpr Variant V_WIDEK96_W8_N2_D96    = make_variant<128,  96, 8, 2, 2,  96, true>("widek96_w8_n2_d96");
constexpr Variant V_WIDEK96_W8_N2_D192   = make_variant<128,  96, 8, 2, 2, 192, true>("widek96_w8_n2_d192");
constexpr Variant V_WIDEK96_W8_N2_D224   = make_variant<128,  96, 8, 2, 2, 224, true>("widek96_w8_n2_d224");

// Wide BK=128 D-specialized (only D=64,96 fit SMEM; D=128 already exists).
constexpr Variant V_WIDEK128_W8_D64      = make_variant<128, 128, 8, 2, 1,  64, true>("widek128_w8_d64");
constexpr Variant V_WIDEK128_W8_D96      = make_variant<128, 128, 8, 2, 1,  96, true>("widek128_w8_d96");

// 3-stage wide BK=64 D-specialized for the larger D values.
constexpr Variant V_WIDE_3_W8_D192       = make_variant<128,  64, 8, 3, 1, 192, true>("wide_3_w8_d192");
constexpr Variant V_WIDE_3_W8_D224       = make_variant<128,  64, 8, 3, 1, 224, true>("wide_3_w8_d224");
constexpr Variant V_WIDE_3_W8_D256       = make_variant<128,  64, 8, 3, 1, 256, true>("wide_3_w8_d256");
constexpr Variant V_WIDE_3_W8_N2_D192    = make_variant<128,  64, 8, 3, 2, 192, true>("wide_3_w8_n2_d192");
constexpr Variant V_WIDE_3_W8_N2_D224    = make_variant<128,  64, 8, 3, 2, 224, true>("wide_3_w8_n2_d224");
constexpr Variant V_WIDE_3_W8_N2_D256    = make_variant<128,  64, 8, 3, 2, 256, true>("wide_3_w8_n2_d256");
```

- [ ] **Step 2: Build to confirm new instantiations compile**

Run: `uv run python setup.py build_ext --inplace`
Expected: clean build; build wall-time should not exceed ~1.5× the pre-Stage-2 number. If it does, drop the BK=128 D=64/96 variants (least value vs cost) and rebuild.

- [ ] **Step 3: Commit**

```bash
git add flash_kmeans_cuda/csrc/assign/assign_policy.cu
git commit -m "Add wide-tile D-specialized variants for D in {64,96,192,224,256}"
```

---

### Task 8: Wire new variants into the per-D policy rows

**Files:**
- Modify: `flash_kmeans_cuda/csrc/assign/assign_policy.cu` (replace the `kD64`/`kD96`/`kD192`/`kD224`/`kD256` rows; D=128 unchanged; D=320/D=384 unchanged)

- [ ] **Step 1: Replace rows with new variant orders**

```cpp
constexpr PolicyRow kD64 = {
  &V_WIDEK128_W8_N2_D64, &V_WIDEK128_W8_D64, &V_WIDEK96_W8_N2_D64,
  &V_WIDEK96_W8_D64, &V_WIDEK128_W8_N2, &V_WIDEK96_W8_N2,
  &V_WIDE_3_W8, &V_NARROW_4,
};

constexpr PolicyRow kD96 = {
  &V_WIDEK128_W8_N2_D96, &V_WIDEK128_W8_D96, &V_WIDEK96_W8_N2_D96,
  &V_WIDEK96_W8_D96, &V_WIDEK128_W8_N2, &V_WIDEK96_W8_N2,
  &V_WIDE_3_W8, &V_NARROW_4,
};

constexpr PolicyRow kD192 = {
  &V_WIDEK96_W8_N2_D192, &V_WIDEK96_W8_D192, &V_WIDE_3_W8_N2_D192,
  &V_WIDE_3_W8_D192, &V_NARROWK32_W4_N2_D192, &V_NARROWK32_W4_N2,
  &V_NARROWK32_W4, &V_NARROW_4,
};

constexpr PolicyRow kD224 = {
  &V_WIDEK96_W8_N2_D224, &V_WIDEK96_W8_D224, &V_WIDE_3_W8_N2_D224,
  &V_WIDE_3_W8_D224, &V_NARROWK32_W4_N2_D224, &V_NARROWK32_W4_N2,
  &V_NARROWK32_W4, &V_NARROW_4,
};

constexpr PolicyRow kD256 = {
  &V_WIDE_3_W8_N2_D256, &V_WIDE_3_W8_D256, &V_NARROWK32_W4_N2_D256,
  &V_NARROWK32_W4_N2, &V_NARROWK32_W4, &V_WIDE_3_W4,
  &V_NARROW_4, &V_DEEP_2_W4,
};
```

(D=128, D=320, D=384, OTHER unchanged.)

- [ ] **Step 2: Build and run equivalence test**

Run: `uv run python setup.py build_ext --inplace && uv run pytest tests/test_assign_dispatch_equiv.py tests/test_correctness.py tests/test_persistent.py -x -q`
Expected: all pass. New variants are correctness-equivalent to existing ones (same kernel template, just different D_FIXED).

- [ ] **Step 3: Commit**

```bash
git add flash_kmeans_cuda/csrc/assign/assign_policy.cu
git commit -m "Promote new D-specialized variants in policy rows"
```

---

### Task 9: Add D-coverage test for all locked D values

**Files:**
- Modify: `tests/test_assign_dispatch_equiv.py` (append a new test)

- [ ] **Step 1: Append the test**

```python
@pytest.mark.parametrize(
    "D",
    [64, 96, 128, 192, 224, 256, 320, 384],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_all_locked_d_values_dispatch(D, dtype):
    """Stage 2 acceptance: every D in the locked set produces correct
    cluster_ids vs the python reference, for both dtypes and a mid-range K."""
    torch.manual_seed(0)
    B, N, K = 1, 2048, 256
    x = torch.randn(B, N, D, device="cuda", dtype=dtype)
    centroids = x[:, :K].contiguous()
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    ids = _C.euclid_assign(x, centroids, x_sq, c_sq, None)
    ref = _assign_safe(x, centroids)
    disagree = (ids != ref).float().mean().item()
    assert disagree < 0.02, (
        f"D={D} dtype={dtype}: {disagree:.3%} disagreement vs reference"
    )
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_assign_dispatch_equiv.py::test_all_locked_d_values_dispatch -v`
Expected: 16 cases (8 D × 2 dtypes) pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_assign_dispatch_equiv.py
git commit -m "Test assign dispatcher across all locked D values"
```

---

## Stage 3 — Autotuner

### Task 10: Add `AutotuneCache` skeleton

**Files:**
- Create: `flash_kmeans_cuda/csrc/assign/assign_autotune.h`
- Create: `flash_kmeans_cuda/csrc/assign/assign_autotune.cu`

- [ ] **Step 1: Write `assign_autotune.h`**

```cpp
// flash_kmeans_cuda/csrc/assign/assign_autotune.h
#pragma once

#include "assign_policy.h"

#include <array>
#include <atomic>
#include <mutex>
#include <span>

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
  std::span<const Variant* const> get_or_probe(
      AutotuneKey key,
      LaunchCtx& ctx,
      std::span<const Variant* const> static_candidates,
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
```

- [ ] **Step 2: Write `assign_autotune.cu` with the probe loop**

```cpp
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

std::span<const Variant* const>
AutotuneCache::get_or_probe(AutotuneKey key,
                             LaunchCtx& ctx,
                             std::span<const Variant* const> static_candidates,
                             bool autotune_enabled,
                             bool verbose) {
  if (!autotune_enabled) return static_candidates;

  Cell& cell = cells_[key.dtype_idx][key.d_idx][key.k_bucket];
  if (cell.probed.load(std::memory_order_acquire)) {
    return std::span<const Variant* const>(cell.ordered.data(), MAX_CAND);
  }

  std::lock_guard<std::mutex> g(probe_mu_);
  if (cell.probed.load(std::memory_order_relaxed)) {
    return std::span<const Variant* const>(cell.ordered.data(), MAX_CAND);
  }

  // Filter SMEM-feasible candidates.
  std::array<const Variant*, MAX_CAND> feasible{};
  int n_feasible = 0;
  for (const Variant* v : static_candidates) {
    if (!v) break;
    if (v->smem(ctx.D, ctx.elt_sz) <= ctx.smem_limit) {
      feasible[n_feasible++] = v;
    }
  }
  if (n_feasible == 0) {
    // Nothing fits — leave cell empty so the dispatcher falls through to the
    // safe kernel.
    cell.probed.store(true, std::memory_order_release);
    return std::span<const Variant* const>(cell.ordered.data(), MAX_CAND);
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
            [](auto& a, auto& b) { return a.first < b.first; });
  int out = 0;
  for (int i = 0; i < n_probe; ++i) cell.ordered[out++] = timings[i].second;
  for (int i = kProbeTopN; i < n_feasible && out < MAX_CAND; ++i) {
    cell.ordered[out++] = feasible[i];
  }
  while (out < MAX_CAND) cell.ordered[out++] = nullptr;

  cell.probed.store(true, std::memory_order_release);
  return std::span<const Variant* const>(cell.ordered.data(), MAX_CAND);
}

AutotuneCache& autotune_cache() {
  static AutotuneCache c;
  return c;
}

}  // namespace assign
}  // namespace fkc
```

- [ ] **Step 3: Add to setup.py source list**

Modify `setup.py` source list:

```python
sources = [
    str(CSRC / "bindings.cpp"),
    str(CSRC / "assign" / "assign_safe.cu"),
    str(CSRC / "assign" / "assign_sm80.cu"),
    str(CSRC / "assign" / "assign_policy.cu"),
    str(CSRC / "assign" / "assign_autotune.cu"),
    str(CSRC / "update" / "update_sorted.cu"),
    str(CSRC / "update" / "update_finalize.cu"),
    str(NB_COMBINED),
]
```

- [ ] **Step 4: Build (autotuner not yet wired into dispatch)**

Run: `uv run python setup.py build_ext --inplace`
Expected: clean build.

- [ ] **Step 5: Commit**

```bash
git add flash_kmeans_cuda/csrc/assign/assign_autotune.h flash_kmeans_cuda/csrc/assign/assign_autotune.cu setup.py
git commit -m "Add AutotuneCache skeleton with cudaEvent-timed probe loop"
```

---

### Task 11: Wire autotuner into `launch_assign_sm80`

**Files:**
- Modify: `flash_kmeans_cuda/csrc/assign/assign_sm80.cu` (the dispatcher)

- [ ] **Step 1: Route through the cache**

Add `#include "assign_autotune.h"` near the top of `assign_sm80.cu`.

In `launch_assign_sm80`, replace the `if (knobs.has_force_override())` block with:

```cpp
  std::span<const Variant* const> candidates;
  int dtype_idx = dtype_index_of(x.scalar_type());
  TORCH_CHECK(dtype_idx >= 0, "assign_sm80 requires fp16 or bf16 input");
  if (knobs.has_force_override()) {
    candidates = build_forced_candidates(knobs, ctx);
  } else {
    AutotuneKey key{ dtype_idx, d_index_of(D), k_bucket_of(K) };
    candidates = autotune_cache().get_or_probe(
        key, ctx,
        static_policy(dtype_idx, key.d_idx, key.k_bucket),
        knobs.autotune, knobs.verbose);
  }
```

- [ ] **Step 2: Build and run all existing tests**

Run: `uv run python setup.py build_ext --inplace && uv run pytest tests/test_assign_dispatch_equiv.py tests/test_correctness.py tests/test_persistent.py tests/test_dtypes.py tests/test_shapes.py tests/test_mma_optin.py -x -q`
Expected: all pass with autotune ON (default). Cold-cell probe adds ~12 ms latency to the *first* call per cell — total test-suite slowdown should be under a second.

- [ ] **Step 3: Commit**

```bash
git add flash_kmeans_cuda/csrc/assign/assign_sm80.cu
git commit -m "Route assign dispatcher through AutotuneCache"
```

---

### Task 12: Stage 3 acceptance — verbose-log probe-count test

**Files:**
- Create: `tests/test_assign_autotune.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_assign_autotune.py
"""Stage 3 acceptance: autotuner probes exactly once per (dtype, D, k_bucket)
cell, never on subsequent calls. Captured via FKC_AUTOTUNE_VERBOSE on stderr."""

from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest


_SCRIPT = r"""
import os, sys
os.environ['FKC_AUTOTUNE_VERBOSE'] = '1'
import torch
from flash_kmeans_cuda import _C

torch.manual_seed(0)

def go(D, K):
    B, N = 1, 2048
    x = torch.randn(B, N, D, device='cuda', dtype=torch.float16)
    centroids = x[:, :K].contiguous()
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    return _C.euclid_assign(x, centroids, x_sq, c_sq, None)

# First sweep: every (D, k_bucket) cell touched here should print probes.
for D in [64, 128, 256]:
    for K in [64, 256, 1024]:
        go(D, K)

print('--- second pass ---', flush=True)

# Second sweep: same shapes -> ZERO new probe lines.
for D in [64, 128, 256]:
    for K in [64, 256, 1024]:
        go(D, K)
"""


def test_autotune_probes_once_per_cell():
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"script failed:\n{proc.stderr}"
    out = proc.stderr
    # Split on the marker.
    parts = out.split("--- second pass ---")
    assert len(parts) == 2, f"missing marker; full stderr:\n{out}"
    first, second = parts
    probe_re = re.compile(r"\[fkc autotune\].* probe\[\d+\]=")
    first_probes = probe_re.findall(first)
    second_probes = probe_re.findall(second)
    # First pass: each of 9 (D, K) cells * up to 3 candidates = up to 27
    # probe lines. Lower bound: each cell prints >= 1 candidate.
    assert len(first_probes) >= 9, (
        f"expected >=9 probes on first pass, got {len(first_probes)}\n{first}"
    )
    # Second pass: zero new probes (cache hits everywhere).
    assert len(second_probes) == 0, (
        f"expected zero probes on second pass, got {len(second_probes)}\n{second}"
    )


def test_autotune_disabled_uses_static_order():
    """FKC_AUTOTUNE=0 must skip the probe entirely (no probe log lines)."""
    script = (
        "import os; os.environ['FKC_AUTOTUNE'] = '0'; "
        "os.environ['FKC_AUTOTUNE_VERBOSE'] = '1'\n"
        "import torch\n"
        "from flash_kmeans_cuda import _C\n"
        "torch.manual_seed(0)\n"
        "x = torch.randn(1, 2048, 128, device='cuda', dtype=torch.float16)\n"
        "centroids = x[:, :256].contiguous()\n"
        "x_sq = (x.float() ** 2).sum(-1).contiguous()\n"
        "c_sq = (centroids.float() ** 2).sum(-1).contiguous()\n"
        "for _ in range(5):\n"
        "    _C.euclid_assign(x, centroids, x_sq, c_sq, None)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "[fkc autotune]" not in proc.stderr, (
        f"expected no probe lines with FKC_AUTOTUNE=0, got:\n{proc.stderr}"
    )
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_assign_autotune.py -v`
Expected: both tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_assign_autotune.py
git commit -m "Test autotuner probes once per cell and respects FKC_AUTOTUNE=0"
```

---

### Task 13: Stage 3 acceptance — concurrent-stream thread-safety stress

**Files:**
- Modify: `tests/test_assign_autotune.py` (append)

- [ ] **Step 1: Append the stress test**

```python
def test_autotune_concurrent_first_call_single_probe():
    """Two host threads launching assign on the same shape concurrently must
    cause exactly one probe (the second thread sees probed=true and reuses).

    Run in a subprocess to keep VERBOSE output clean.
    """
    script = r"""
import os
os.environ['FKC_AUTOTUNE_VERBOSE'] = '1'
import threading
import torch
from flash_kmeans_cuda import _C

torch.manual_seed(0)
B, N, K, D = 1, 2048, 256, 128
x = torch.randn(B, N, D, device='cuda', dtype=torch.float16)
centroids = x[:, :K].contiguous()
x_sq = (x.float() ** 2).sum(-1).contiguous()
c_sq = (centroids.float() ** 2).sum(-1).contiguous()

barrier = threading.Barrier(4)

def worker():
    barrier.wait()
    for _ in range(3):
        _C.euclid_assign(x, centroids, x_sq, c_sq, None)

threads = [threading.Thread(target=worker) for _ in range(4)]
for t in threads: t.start()
for t in threads: t.join()
torch.cuda.synchronize()
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    # Expect probe lines for ONE (D=128, k_bucket=2) cell only — exactly
    # min(3, n_feasible) lines, which for D=128 mid-K is 3.
    probes = re.findall(r"\[fkc autotune\].* probe\[\d+\]=", proc.stderr)
    cells = set(
        re.findall(r"D_idx=(\d+) k_bucket=(\d+)", proc.stderr)
    )
    assert len(cells) == 1, (
        f"expected exactly 1 cell probed under concurrent load, got "
        f"{len(cells)} cells: {cells}\n{proc.stderr}"
    )
    assert 1 <= len(probes) <= 3, (
        f"expected 1-3 probe lines for one cell, got {len(probes)}:\n{proc.stderr}"
    )
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_assign_autotune.py::test_autotune_concurrent_first_call_single_probe -v`
Expected: pass. The double-checked-lock + atomic in `AutotuneCache::get_or_probe` ensures the second thread to acquire `probe_mu_` sees `probed=true` and returns without re-probing.

- [ ] **Step 3: Commit**

```bash
git add tests/test_assign_autotune.py
git commit -m "Stress autotuner cache under concurrent host threads"
```

---

### Task 14: Final sweep — run the full suite + the assign benchmark

- [ ] **Step 1: Full pytest run**

Run: `uv run pytest tests/ -x -q`
Expected: all pass.

- [ ] **Step 2: Compare benchmarks against `main`**

```bash
git stash --keep-index   # if any uncommitted leftovers
uv run python benchmarks/bench_d_sweep.py | tee /tmp/bench_after.txt
git checkout main -- benchmarks/bench_d_sweep.py  # ensure same script
# Re-run on `main` for a baseline only if the user wants raw numbers; otherwise
# just confirm bench_after has no NaN / errors.
```

Expected: every (D, K) cell completes; no NaN; D=128 perf is within ±2 % of pre-refactor (Stage 1 acceptance); D=192 / D=256 mega-K cells improve measurably (Stage 2 acceptance) thanks to the new wide-tile + raw-distance specializations now seeded ahead of `narrowk32_w4_n2`.

- [ ] **Step 3: Final commit (if any benchmark-driven fix needed)**

If any cell regresses, reorder its row in `assign_policy.cu` and commit:

```bash
git add flash_kmeans_cuda/csrc/assign/assign_policy.cu
git commit -m "Reorder policy row for <D>/<K_bucket> based on bench results"
```

If no regression, no commit needed.

---

## Self-Review Notes

- **Spec coverage:** All four locked decisions in the spec map to tasks: D set → Task 7+8 catalog/rows; K-buckets → `k_bucket_of` in Task 3; autotune key → `AutotuneKey` in Task 10; probe budget → `kProbeTopN`/`kProbeIters` constants in Task 10. Stage 1 acceptance → Task 5; Stage 2 acceptance → Task 9 + Task 14 bench check; Stage 3 acceptance → Task 12 + Task 13. Env-knob per-call gap → Task 3 (`read_env_knobs`) + Task 4 (called every entry) + Task 6 (test simplification confirms it works).
- **Placeholder scan:** No "TBD" / "implement later" / "similar to Task N" anywhere — every code step shows the actual code; every test step shows the actual test.
- **Type consistency:** `Variant`, `LaunchCtx`, `AutotuneKey`, `EnvKnobs`, `PolicyRow`, `MAX_CAND`, `N_DTYPES`, `N_D_IDX`, `N_K_BUCKET` are defined once (in `assign_variants.h` / `assign_policy.h`) and referenced consistently in the catalog (`Task 3`), the dispatcher rewrite (`Task 4`), and the autotuner (`Task 10`). `dtype_index_of`, `d_index_of`, `k_bucket_of`, `read_env_knobs`, `static_policy`, `build_forced_candidates`, `autotune_cache`, `AutotuneCache::get_or_probe` signatures match between header declaration and `.cu` definition.
- **Risk acknowledged in spec:** Task 7's commit message and step 2 explicitly call out the "drop BK=128 D=64/96 if compile-time blows up" mitigation from the spec's open-risks section.
