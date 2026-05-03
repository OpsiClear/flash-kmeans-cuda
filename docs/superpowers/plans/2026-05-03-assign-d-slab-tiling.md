# Assign D-Slab Tiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a parallel `assign_sm80_dslab_kernel` template that holds one D-slab in SMEM at a time so D ∈ {192, 224, 256, 320, 384} can run the BN=128 BK=128 STAGES=2 8-warp tile that hits 202 TFLOPS at D=128 today. Target: D=192/256 mega-K from 131/149 TFLOPS to ≥180 TFLOPS.

**Architecture:** New kernel template + variant factory + 10 catalog entries (5 D values × 2 N_TILES). Drops into the existing `Variant` policy table and `AutotuneCache` from the policy-restructure work — no dispatcher changes needed. Kernel is built by cloning the legacy `assign_sm80_kernel`, adding `D_FULL`/`SLAB_WIDTHS`/`SMEM_PAD_SLAB` template params, then refactoring the K-chunk inner loop to iterate slabs and accumulate cross-products across slabs.

**Tech Stack:** CUDA / nvcc 12+, ATen / libtorch, nanobind extension, pytest, `setup.py` (CUDAExtension).

**Source spec:** `docs/superpowers/specs/2026-05-03-assign-d-slab-tiling-design.md` (commit `8708ff7`).

---

## File Map

| Path                                                          | Action  | Responsibility                                                                                                  |
|---------------------------------------------------------------|---------|-----------------------------------------------------------------------------------------------------------------|
| `flash_kmeans_cuda/csrc/assign/assign_sm80_dslab_kernel.cuh`  | Create  | New kernel template `assign_sm80_dslab_kernel` + its anonymous-namespace device helpers.                        |
| `flash_kmeans_cuda/csrc/assign/assign_dslab_variants.h`       | Create  | `DSlabVariantSpec<>`, `make_dslab_variant<>`, forward decl of the dslab kernel template.                        |
| `flash_kmeans_cuda/csrc/assign/assign_policy.cu`              | Modify  | Add `kPart192…kPart384` partitions, 10 `V_DSLAB_*` catalog entries, update `kD192/kD224/kD256/kD320/kD384` rows. |
| `flash_kmeans_cuda/csrc/assign/assign_policy.h`               | Modify  | Add `FKC_DSLAB` knob to `EnvKnobs`; declare `build_dslab_force_candidates` for force-route test.                |
| `tests/test_assign_dslab.py`                                  | Create  | Smoke test (vs Python fp32 reference) + cross-kernel agreement test.                                            |

The legacy `assign_sm80_kernel.cuh` is **not modified** — the new kernel is parallel infrastructure.

---

## Task 1: Add `DSlabVariantSpec` factory header

**Files:**
- Create: `flash_kmeans_cuda/csrc/assign/assign_dslab_variants.h`

The factory mirrors `make_variant<>` but is parameterized by `D_FULL` and `SLAB_WIDTHS`. The kernel template is forward-declared here so `make_dslab_variant` can take its address; the definition lands in Task 2.

- [ ] **Step 1: Verify clean tree**

Run: `git status`
Expected: clean (only `.uv-run-env/` untracked).

- [ ] **Step 2: Write `assign_dslab_variants.h`**

Create `flash_kmeans_cuda/csrc/assign/assign_dslab_variants.h`:

```cpp
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
          int S0, int S1, int S2, int S3,         // partition unpacked (NTTP friendly across nvcc versions)
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
    // D-independent: x_slab + STAGES * c_slab + STAGES * c_sq.
    size_t row = (size_t)(SLAB_MAX + SMEM_PAD_SLAB);
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
```

Note: the partition is unpacked into four `int` template parameters (`S0..S3`) rather than passing a `std::array<int,4>` as a single non-type template parameter. NVCC's NTTP-class-type support is uneven across versions; four ints is bulletproof.

- [ ] **Step 3: Sanity-build**

The header isn't included anywhere yet, so the build won't parse it. Add a temporary `#include "assign_dslab_variants.h"` at the top of `assign_policy.cu` (after the existing `#include "assign_policy.h"`), build, then REMOVE the temporary include before committing.

Run: `uv run python setup.py build_ext --inplace`
Expected: clean build.

After the build succeeds, remove the temporary include from `assign_policy.cu`.

- [ ] **Step 4: Commit**

```bash
git add flash_kmeans_cuda/csrc/assign/assign_dslab_variants.h
git commit -m "Add D-slab variant factory header"
```

Verify: `git show --stat HEAD` shows only the new header.

---

## Task 2: Clone the legacy kernel as `assign_sm80_dslab_kernel.cuh`

Create the new kernel file as a literal copy of `assign_sm80_kernel.cuh`, with the kernel renamed and the new template parameters added (but unused at this stage). The kernel still loads full x_smem and full c_tile per K-chunk just like the legacy. This task is pure scaffolding — it proves the wiring (catalog → factory → kernel) end-to-end without changing kernel behavior.

**Files:**
- Create: `flash_kmeans_cuda/csrc/assign/assign_sm80_dslab_kernel.cuh`

- [ ] **Step 1: Copy the legacy kernel**

Run:
```bash
cp flash_kmeans_cuda/csrc/assign/assign_sm80_kernel.cuh flash_kmeans_cuda/csrc/assign/assign_sm80_dslab_kernel.cuh
```

- [ ] **Step 2: Rename the kernel and add new template params**

In `flash_kmeans_cuda/csrc/assign/assign_sm80_dslab_kernel.cuh`:

a. **At the top of the file** (line ~1, the file-level comment), replace the legacy comment block with:

```cpp
// Ampere+ (sm_80) D-slab variant of the assign kernel. Holds one D-slab
// (max 128 features) in SMEM at a time, with the K-chunk inner loop
// iterating slabs and accumulating cross-products in registers across
// slabs. SMEM footprint is D-independent at SLAB_MAX=128, so every
// target D ∈ {192, 224, 256, 320, 384} fits the BN=128 BK=128 STAGES=2
// 8-warp tile that hits 202 TFLOPS at D=128 in the legacy kernel.
//
// Cloned from assign_sm80_kernel.cuh in Task 2 of the D-slab plan; the
// inner loop is restructured in Task 4. Until Task 4, this kernel is
// behaviorally identical to the legacy at supported D values.
```

b. **Replace the kernel template signature and the IMPORTANT-comment block.** Find the legacy template starting at "template <typename T, int BLOCK_N, int BLOCK_K, int WARPS_PER_CTA, int PIPE_STAGES,..." (around line 146 in the legacy). Replace:

Old (legacy):
```cpp
template <typename T, int BLOCK_N, int BLOCK_K, int WARPS_PER_CTA, int PIPE_STAGES,
          int N_TILES_PER_CTA = 1, bool ASYNC_CSQ = false, int D_FIXED = 0,
          bool RAW_DIST = false>
__global__ void __launch_bounds__(WARPS_PER_CTA * 32, 1)
assign_sm80_kernel(
```

New (dslab):
```cpp
template <typename T,
          int BLOCK_N, int BLOCK_K, int WARPS_PER_CTA, int PIPE_STAGES,
          int N_TILES_PER_CTA,
          int D_FULL,
          int S0, int S1, int S2, int S3,
          int SMEM_PAD_SLAB,
          bool RAW_DIST>
__global__ void __launch_bounds__(WARPS_PER_CTA * 32, 1)
assign_sm80_dslab_kernel(
```

c. **Inside the kernel body**, near the top (just after the `static_assert`s), add a `D_TILE` definition and a compile-time `constexpr` partition array:

```cpp
  static_assert(D_FULL > 0, "D-slab kernel requires D_FULL");
  static_assert(N_TILES_PER_CTA >= 1, "N_TILES_PER_CTA must be >= 1");

  // Compile-time slab partition.
  constexpr int kSlabWidths[MAX_SLABS] = {S0, S1, S2, S3};
  constexpr int kNumSlabs =
      (S3 > 0) ? 4 :
      (S2 > 0) ? 3 :
      (S1 > 0) ? 2 :
      (S0 > 0) ? 1 : 0;
  static_assert(kNumSlabs > 0, "D-slab kernel requires non-empty partition");

  // Until Task 4, treat the dslab kernel as legacy with D_TILE = D_FULL.
  const int D_TILE = D_FULL;
  const int D_SMEM = D_TILE + SMEM_PAD;  // legacy SMEM_PAD; Task 6 will switch to SMEM_PAD_SLAB
  const int pid_b = blockIdx.y;
  // ... (rest of the legacy body unchanged)
```

(`MAX_SLABS` is `4`; defined either in `assign_dslab_variants.h` or hoist to a shared constant. Reference `assign_dslab_variants.h`'s constant; this kernel header includes it transitively via `assign_policy.h` once the catalog entry lands.)

d. **Rest of the body:** leave unchanged. The dslab kernel's body is identical to the legacy at this point — same K-chunk loop, same x_smem load, same c_tile pipeline, same epilogue.

e. **Anonymous-namespace device helpers** (`mma_atom`, `async_load_tile`, `async_load_csq_full_tile`, `store_csq_tile`): keep them in this file's anonymous namespace just like the legacy. They get duplicated across TUs that include both .cuh files; nvcc inlines them so the binary cost is zero.

- [ ] **Step 3: Add the include path**

The kernel will be reached by including this header from `assign_dslab_variants.h` indirectly (through the catalog wire-up in Task 3). For now, just verify the header parses by adding `#include "assign_sm80_dslab_kernel.cuh"` temporarily at the top of `assign_policy.cu`.

Run: `uv run python setup.py build_ext --inplace`
Expected: clean build (kernel template is declared but not yet instantiated; no instantiation cost).

Remove the temporary include from `assign_policy.cu`.

- [ ] **Step 4: Commit**

```bash
git add flash_kmeans_cuda/csrc/assign/assign_sm80_dslab_kernel.cuh
git commit -m "Clone assign_sm80_kernel as dslab kernel scaffold"
```

Verify: `git show --stat HEAD` shows only the new file.

---

## Task 3: Add `V_DSLAB_W8_N2_D256` catalog entry + `FKC_DSLAB` force-route knob

D=256 is the simplest case (uniform partition `[128, 128]`). Add ONE catalog entry plus a force-route environment knob so a smoke test can target it directly without going through the autotuner.

**Files:**
- Modify: `flash_kmeans_cuda/csrc/assign/assign_policy.h` (add `FKC_DSLAB` knob to `EnvKnobs`)
- Modify: `flash_kmeans_cuda/csrc/assign/assign_policy.cu` (catalog entry + force builder + read_env_knobs update)

- [ ] **Step 1: Extend `EnvKnobs`**

In `flash_kmeans_cuda/csrc/assign/assign_policy.h`, add to the `EnvKnobs` struct (alongside existing `wide3`, `w4`, etc.):

```cpp
  bool dslab            = false; // FKC_DSLAB — force-route to D-slab variant
```

And in `has_force_override()`:

```cpp
  bool has_force_override() const {
    return wide3 || w4 || narrow || deep || dslab || n_tiles_override != 0;
  }
```

- [ ] **Step 2: Add catalog entry and force-row in `assign_policy.cu`**

After the existing Stage 2 catalog block (search for `// === Stage 2 additions ===`), append:

```cpp
// === D-slab variants ======================================================
// Per-D template instantiations with compile-time partition baked in.
constexpr Variant V_DSLAB_W8_N2_D256 =
    make_dslab_variant<128, 128, 8, 2, 2, 256, 128, 128, 0, 0>("dslab_w8_n2_d256");
constexpr Variant V_DSLAB_W8_D256 =
    make_dslab_variant<128, 128, 8, 2, 1, 256, 128, 128, 0, 0>("dslab_w8_d256");
```

Add the include at the top of `assign_policy.cu`:

```cpp
#include "assign_dslab_variants.h"
```

Add a forced-row for FKC_DSLAB in the anonymous namespace where `kForcedDeep` etc. live:

```cpp
constexpr PolicyRow kForcedDslabD256 = {
  &V_DSLAB_W8_N2_D256, &V_DSLAB_W8_D256, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
};
```

- [ ] **Step 3: Wire `FKC_DSLAB` into `read_env_knobs` and `build_forced_candidates`**

In `read_env_knobs()`:
```cpp
  k.dslab    = truthy(std::getenv("FKC_DSLAB"));
```

In `build_forced_candidates()`, add a branch (place it ABOVE the existing `if (knobs.deep)` so dslab takes precedence — this is a developer override knob):

```cpp
  if (knobs.dslab) {
    // Per-D dslab force-row. For Task 3, only D=256 is wired; later tasks add
    // D=192/224/320/384.
    if (ctx.D == 256) {
      return VariantView(kForcedDslabD256.data(), MAX_CAND);
    }
    // Unknown D for FKC_DSLAB — fall through to safe kernel via empty row.
    static constexpr PolicyRow kEmpty = {nullptr, nullptr, nullptr, nullptr,
                                          nullptr, nullptr, nullptr, nullptr};
    return VariantView(kEmpty.data(), MAX_CAND);
  }
```

- [ ] **Step 4: Build**

Run: `uv run python setup.py build_ext --inplace`
Expected: clean build. The dslab kernel template is now instantiated for D=256 fp16 + bf16, NT=1 + NT=2 = 4 instantiations.

- [ ] **Step 5: Commit**

```bash
git add flash_kmeans_cuda/csrc/assign/assign_policy.h flash_kmeans_cuda/csrc/assign/assign_policy.cu
git commit -m "Add FKC_DSLAB knob and V_DSLAB_*_D256 catalog entries"
```

---

## Task 4: Smoke test for D=256 dslab path (still legacy-equivalent)

Verify the cloned kernel produces correct cluster_ids when force-routed via `FKC_DSLAB=1`. At this stage the dslab kernel is functionally identical to the legacy (no slab loop yet), so the test just proves the wiring works.

**Files:**
- Create: `tests/test_assign_dslab.py`

- [ ] **Step 1: Write the smoke test**

```python
# tests/test_assign_dslab.py
"""D-slab kernel correctness tests.

The smoke test (test_dslab_smoke_d256) confirms the new kernel produces correct
cluster_ids when force-routed via FKC_DSLAB=1. After Task 5 (slab inner loop),
the cross-kernel test (test_dslab_matches_narrow) confirms equivalence to the
legacy narrowk32 path for every supported D.

Each test runs in a subprocess so FKC_DSLAB takes effect at first launch."""

from __future__ import annotations

import subprocess
import sys

import pytest


def _run_dslab_script(D: int, K: int, dtype: str = "float16") -> tuple[int, str]:
    script = f"""
import os
os.environ['FKC_DSLAB'] = '1'
import torch
from flash_kmeans_cuda import _C

torch.manual_seed(0)
B, N, D, K = 1, 2048, {D}, {K}
dtype = torch.{dtype}
x = torch.randn(B, N, D, device='cuda', dtype=dtype)
centroids = x[:, :K].contiguous()
x_sq = (x.float() ** 2).sum(-1).contiguous()
c_sq = (centroids.float() ** 2).sum(-1).contiguous()

ids = _C.euclid_assign(x, centroids, x_sq, c_sq, None)

# Python fp32 reference.
diff = x.unsqueeze(2).float() - centroids.unsqueeze(1).float()
ref = (diff * diff).sum(-1).argmin(dim=-1).to(torch.int32)

disagree = (ids != ref).float().mean().item()
threshold = 0.05 if dtype == torch.bfloat16 else 0.02
assert disagree < threshold, (
    f"D={{D}} K={{K}} dtype={{dtype}}: {{disagree:.3%}} disagreement "
    f"(threshold {{threshold:.0%}})"
)
print(f'OK D={{D}} K={{K}} disagree={{disagree:.4%}}')
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120,
    )
    return proc.returncode, proc.stdout + proc.stderr


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_dslab_smoke_d256(dtype):
    """D=256 force-routed via FKC_DSLAB=1 produces correct cluster_ids."""
    rc, out = _run_dslab_script(D=256, K=256, dtype=dtype)
    assert rc == 0, f"D=256 dslab smoke failed:\n{out}"
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_assign_dslab.py::test_dslab_smoke_d256 -v`
Expected: 2 cases (fp16 + bf16) pass.

If a case fails, STOP and report as BLOCKED. The dslab kernel at this stage is supposed to be byte-identical-behavior to the legacy at D=256, so a failure means the clone-and-rename in Task 2 introduced a bug.

- [ ] **Step 3: Run the full test suite to confirm no regression**

Run: `uv run pytest tests/test_assign_dispatch_equiv.py tests/test_correctness.py tests/test_persistent.py tests/test_assign_autotune.py -x -q`
Expected: all pass. The dslab catalog entries are reachable only via FKC_DSLAB=1, so default-path behavior is unchanged.

- [ ] **Step 4: Commit**

```bash
git add tests/test_assign_dslab.py
git commit -m "Add D-slab smoke test for D=256 (force-routed via FKC_DSLAB)"
```

---

## Task 5: Implement the D-slab inner loop

This is the substantive kernel rewrite. Modify `assign_sm80_dslab_kernel` to:
1. Allocate x_smem and c_smem at `SLAB_MAX` width (not `D_TILE`).
2. Restructure the K-chunk inner loop to iterate over slabs (per `kSlabWidths[]`).
3. Accumulate `partial_cross[m][n]` across slabs within each K-chunk.
4. Switch `SMEM_PAD` references to `SMEM_PAD_SLAB` for the slab buffers.

**Files:**
- Modify: `flash_kmeans_cuda/csrc/assign/assign_sm80_dslab_kernel.cuh` (kernel body)

The legacy kernel's structure (after Task 2's changes) is:

```
for n_tile in 0..N_TILES_PER_CTA:
    init best[m] = {+INF, -1}
    if n_tile == 0:
        async_load_tile x_smem  <- x[n_start..n_start+BLOCK_N, :D]    # full D
        cp_async commit
    for k_chunk in 0..K step BLOCK_K:
        async_load_tile c_smem[stage]  <- centroids[k_chunk, :D]      # full D
        async_load_csq c_sq_smem[stage] <- c_sq[k_chunk]
        cp_async commit; wait_group<PIPE_STAGES-1>; sync
        for d_step in 0..D step BLOCK_D=16:
            for m, n: mma(x_smem[m, d_step], c_smem[stage, n, d_step], cross[m][n])
        // Epilogue: dist + best update.
        for m, n in regs: dist = ...; update best[m]
    if N_TILES_PER_CTA > 1:
        async_load_tile x_smem  <- x[next n_tile, :D]                 # prefetch next
        cp_async commit
    write cluster_ids[n_tile_rows] = best[:].idx
```

The new structure is:

```
for n_tile in 0..N_TILES_PER_CTA:
    init best[m] = {+INF, -1}
    for k_chunk in 0..K step BLOCK_K:
        init partial_cross[m][n] = 0
        slab_off = 0
        for slab_idx in 0..kNumSlabs:                              # unrolled
            slab_w = kSlabWidths[slab_idx]
            async_load_tile x_slab_smem  <- x[n_start..n_start+BLOCK_N, slab_off..slab_off+slab_w]
            async_load_tile c_slab_smem[stage] <- centroids[k_chunk, slab_off..slab_off+slab_w]
            if slab_idx == 0:
                async_load_csq c_sq_smem[stage] <- c_sq[k_chunk]
            cp_async commit; wait_all; sync
            for d_step in 0..slab_w step BLOCK_D=16:               # unrolled
                for m, n: mma(x_slab_smem[m, d_step], c_slab_smem[stage, n, d_step], partial_cross[m][n])
            slab_off += slab_w
        // Epilogue (same as before but uses partial_cross instead of cross).
        for m, n in regs: dist = ...; update best[m]
    write cluster_ids[n_tile_rows] = best[:].idx
```

Key differences:
- **No persistent x_smem** — loaded per K-chunk per slab.
- **No PIPE_STAGES rotation on x** — x_slab is single-buffered (one CTA-shared buffer per slab iteration).
- **PIPE_STAGES rotation on c** — c_slab still multi-stage (between K-chunks); the per-slab loads within a K-chunk use the SAME stage.
- **`xs_top_cache` / `xs_bot_cache` in legacy** — these registers cached per-warp x_sq fragments. Replace with simpler slab-local register state.

This is a substantial restructure. Approach it as a series of small edits:

- [ ] **Step 1: Update the SMEM layout block**

Find the SMEM allocation block in the kernel (after the kernel body opens, around the `extern __shared__ unsigned char smem_raw[];` line). Replace:

Old:
```cpp
  const int D_TILE = D_FULL;
  const int D_SMEM = D_TILE + SMEM_PAD;
  extern __shared__ unsigned char smem_raw[];
  T* x_smem = reinterpret_cast<T*>(smem_raw);
  T* c_smem = x_smem + (size_t)BLOCK_N * D_SMEM;
  float* c_sq_smem = reinterpret_cast<float*>(
      c_smem + (size_t)PIPE_STAGES * BLOCK_K * D_SMEM);
```

New:
```cpp
  // D-slab SMEM layout: x_slab[BN, SLAB_MAX+PAD] + c_slab[PIPE, BK, SLAB_MAX+PAD] + c_sq[PIPE, BK]
  constexpr int D_SLAB_SMEM = SLAB_MAX + SMEM_PAD_SLAB;
  extern __shared__ unsigned char smem_raw[];
  T* x_slab_smem = reinterpret_cast<T*>(smem_raw);
  T* c_slab_smem = x_slab_smem + (size_t)BLOCK_N * D_SLAB_SMEM;
  float* c_sq_smem = reinterpret_cast<float*>(
      c_slab_smem + (size_t)PIPE_STAGES * BLOCK_K * D_SLAB_SMEM);
```

- [ ] **Step 2: Remove the `n_tile == 0` x_smem prefetch**

Find the block in the n_tile loop:
```cpp
    if (n_tile == 0) {
      async_load_tile<T, THREADS_PER_CTA>(x_smem,
                         x + (size_t)pid_b * N * D_TILE + (size_t)n_start * D_TILE,
                         n_count, BLOCK_N, D_TILE, D_SMEM);
      ptx::cp_async_commit();
    }
```
Delete it. x is now loaded per-K-chunk-per-slab.

Also find and DELETE the matching tail prefetch at the end of the n_tile loop (before `write cluster_ids`):
```cpp
    if (N_TILES_PER_CTA > 1 && (n_tile + 1) < N_TILES_PER_CTA) {
      // ... prefetches next n_tile's x_smem
    }
```

- [ ] **Step 3: Restructure the K-chunk loop body**

Find the K-chunk loop (`for (int k_chunk = 0; k_chunk < num_k_chunks; ++k_chunk)`). Replace its body to:

```cpp
    for (int k_chunk = 0; k_chunk < num_k_chunks; ++k_chunk) {
      const int k_start = k_chunk * BLOCK_K;
      const int k_count = min(BLOCK_K, K - k_start);
      const int stage = k_chunk % PIPE_STAGES;

      // Per-K-chunk register state: partial_cross[m][n] accumulates across slabs.
      float partial_cross[M_ATOMS_PER_WARP * 2][N_ATOMS_PER_WARP] = {{0.f}};

      int slab_off = 0;
      #pragma unroll
      for (int slab_idx = 0; slab_idx < kNumSlabs; ++slab_idx) {
        const int slab_w = kSlabWidths[slab_idx];

        // Load x_slab and c_slab (and c_sq on slab 0 only).
        async_load_tile<T, THREADS_PER_CTA>(
            x_slab_smem,
            x + (size_t)pid_b * N * D_FULL + (size_t)n_start * D_FULL + slab_off,
            n_count, BLOCK_N, slab_w, D_SLAB_SMEM);
        async_load_tile<T, THREADS_PER_CTA>(
            c_slab_smem + (size_t)stage * BLOCK_K * D_SLAB_SMEM,
            centroids + (size_t)pid_b * K * D_FULL + (size_t)k_start * D_FULL + slab_off,
            k_count, BLOCK_K, slab_w, D_SLAB_SMEM);
        if (slab_idx == 0) {
          async_load_csq_full_tile<THREADS_PER_CTA>(
              c_sq_smem + (size_t)stage * BLOCK_K,
              c_sq + (size_t)pid_b * K + k_start, k_count, BLOCK_K);
        }
        ptx::cp_async_commit();
        ptx::cp_async_wait_all();
        __syncthreads();

        // mma loop over this slab's d_steps.
        #pragma unroll
        for (int d_step = 0; d_step < slab_w; d_step += BLOCK_D) {
          #pragma unroll
          for (int m_atom = 0; m_atom < M_ATOMS_PER_WARP; ++m_atom) {
            #pragma unroll
            for (int n_atom = 0; n_atom < N_ATOMS_PER_WARP; ++n_atom) {
              mma_atom<T>(
                  x_slab_smem, c_slab_smem + (size_t)stage * BLOCK_K * D_SLAB_SMEM,
                  /*x_smem_stride=*/D_SLAB_SMEM,
                  /*c_smem_stride=*/D_SLAB_SMEM,
                  /*x_row_base=*/warp_id * WARP_M + m_atom * 16,
                  /*c_row_base=*/n_atom * 8,
                  /*d_off=*/d_step,
                  /*lane=*/lane,
                  /*ldm_row_off=*/ldm_row_off,
                  /*ldm_col_off=*/ldm_col_off,
                  /*ldm_row_in_half=*/ldm_row_in_half,
                  /*ldm_n_atom_off=*/ldm_n_atom_off,
                  /*partial_cross=*/partial_cross[m_atom * 2],   // 2 M-halves per atom
                  /*partial_cross_bot=*/partial_cross[m_atom * 2 + 1]);
            }
          }
        }
        slab_off += slab_w;
      }

      // K-chunk epilogue: convert partial_cross to dist and update best[m].
      // (Identical to the legacy kernel's epilogue; only the variable name
      // changes from `cross` to `partial_cross`.)
      #pragma unroll
      for (int m_atom = 0; m_atom < M_ATOMS_PER_WARP; ++m_atom) {
        #pragma unroll
        for (int n_atom = 0; n_atom < N_ATOMS_PER_WARP; ++n_atom) {
          // ... (copy the legacy 4-fp32-reg dist update, replacing `cross[m][n]`
          //      references with `partial_cross[m][n]`)
        }
      }
    }  // k_chunk loop
```

The exact `mma_atom` interface and the epilogue's 4-fp32-reg layout are taken from the legacy kernel — copy them verbatim, just substituting `partial_cross` for the legacy's per-K-chunk `cross` accumulator. The legacy's `xs_top_cache` / `xs_bot_cache` (per-warp x_sq fragment caches) move from CTA-scope to **K-chunk-scope** since x_sq doesn't change per slab — load them once at the top of the K-chunk before the slab loop.

If the `mma_atom` helper signature in `assign_sm80_kernel.cuh` doesn't accept a `partial_cross` parameter directly (the legacy uses inline `asm volatile("mma.sync...")` rather than a helper function), then the inline asm pattern is what you copy — same instruction sequence, different accumulator register.

- [ ] **Step 4: Build**

Run: `uv run python setup.py build_ext --inplace`
Expected: clean build. The dslab kernel now has substantively different SMEM and inner-loop structure.

If the build fails due to register allocation or smem-bytes errors, STOP and report as BLOCKED with the exact compiler error.

- [ ] **Step 5: Run the smoke test**

Run: `uv run pytest tests/test_assign_dslab.py::test_dslab_smoke_d256 -v`
Expected: 2 cases pass (fp16 + bf16).

If a case fails:
- Check the disagreement fraction. If <30% → likely a partial_cross accumulation bug (e.g., missed slab in the carry, off-by-one on `slab_off`). If 100% disagreement on every point → SMEM layout corruption. Both cases STOP and report as BLOCKED.

- [ ] **Step 6: Run the full test suite to confirm no regression**

Run: `uv run pytest tests/test_assign_dispatch_equiv.py tests/test_correctness.py tests/test_persistent.py tests/test_assign_autotune.py -x -q`
Expected: all pass. The dslab kernel is reachable only via FKC_DSLAB=1, so default-path behavior is unchanged.

- [ ] **Step 7: Commit**

```bash
git add flash_kmeans_cuda/csrc/assign/assign_sm80_dslab_kernel.cuh
git commit -m "Implement D-slab inner loop (per-slab SMEM staging + cross carry)"
```

---

## Task 6: Add catalog entries for D ∈ {192, 224, 320, 384}

D=256's smoke test passing in Task 5 proves the slab loop is correct for the uniform `[128, 128]` case. Now add the irregular partitions: `[128, 64]`, `[128, 96]`, `[128, 128, 64]`, `[128, 128, 128]`.

**Files:**
- Modify: `flash_kmeans_cuda/csrc/assign/assign_policy.cu`

- [ ] **Step 1: Add the 8 new catalog entries**

After `V_DSLAB_W8_D256` in the D-slab variants block of `assign_policy.cu`, append:

```cpp
constexpr Variant V_DSLAB_W8_N2_D192 =
    make_dslab_variant<128, 128, 8, 2, 2, 192, 128, 64, 0, 0>("dslab_w8_n2_d192");
constexpr Variant V_DSLAB_W8_D192 =
    make_dslab_variant<128, 128, 8, 2, 1, 192, 128, 64, 0, 0>("dslab_w8_d192");

constexpr Variant V_DSLAB_W8_N2_D224 =
    make_dslab_variant<128, 128, 8, 2, 2, 224, 128, 96, 0, 0>("dslab_w8_n2_d224");
constexpr Variant V_DSLAB_W8_D224 =
    make_dslab_variant<128, 128, 8, 2, 1, 224, 128, 96, 0, 0>("dslab_w8_d224");

constexpr Variant V_DSLAB_W8_N2_D320 =
    make_dslab_variant<128, 128, 8, 2, 2, 320, 128, 128, 64, 0>("dslab_w8_n2_d320");
constexpr Variant V_DSLAB_W8_D320 =
    make_dslab_variant<128, 128, 8, 2, 1, 320, 128, 128, 64, 0>("dslab_w8_d320");

constexpr Variant V_DSLAB_W8_N2_D384 =
    make_dslab_variant<128, 128, 8, 2, 2, 384, 128, 128, 128, 0>("dslab_w8_n2_d384");
constexpr Variant V_DSLAB_W8_D384 =
    make_dslab_variant<128, 128, 8, 2, 1, 384, 128, 128, 128, 0>("dslab_w8_d384");
```

- [ ] **Step 2: Add per-D forced rows for FKC_DSLAB**

In the anonymous namespace where `kForcedDslabD256` lives, add:

```cpp
constexpr PolicyRow kForcedDslabD192 = {
  &V_DSLAB_W8_N2_D192, &V_DSLAB_W8_D192, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
};
constexpr PolicyRow kForcedDslabD224 = {
  &V_DSLAB_W8_N2_D224, &V_DSLAB_W8_D224, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
};
constexpr PolicyRow kForcedDslabD320 = {
  &V_DSLAB_W8_N2_D320, &V_DSLAB_W8_D320, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
};
constexpr PolicyRow kForcedDslabD384 = {
  &V_DSLAB_W8_N2_D384, &V_DSLAB_W8_D384, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
};
```

- [ ] **Step 3: Extend `build_forced_candidates`'s D switch**

Replace the existing `if (knobs.dslab) { if (ctx.D == 256) ...; }` block with:

```cpp
  if (knobs.dslab) {
    switch (ctx.D) {
      case 192: return VariantView(kForcedDslabD192.data(), MAX_CAND);
      case 224: return VariantView(kForcedDslabD224.data(), MAX_CAND);
      case 256: return VariantView(kForcedDslabD256.data(), MAX_CAND);
      case 320: return VariantView(kForcedDslabD320.data(), MAX_CAND);
      case 384: return VariantView(kForcedDslabD384.data(), MAX_CAND);
      default: {
        static constexpr PolicyRow kEmpty = {nullptr, nullptr, nullptr, nullptr,
                                              nullptr, nullptr, nullptr, nullptr};
        return VariantView(kEmpty.data(), MAX_CAND);
      }
    }
  }
```

- [ ] **Step 4: Build**

Run: `uv run python setup.py build_ext --inplace`
Expected: clean build. 16 new kernel instantiations added (8 entries × 2 dtypes). Build wall-time should grow by ~10-20%.

- [ ] **Step 5: Commit**

```bash
git add flash_kmeans_cuda/csrc/assign/assign_policy.cu
git commit -m "Add D-slab catalog entries for D in {192,224,320,384}"
```

---

## Task 7: Cross-kernel agreement test for all 5 D values

Verify the D-slab kernel agrees with the legacy narrowk32 kernel on the same input (with bounded drift for cross-product accumulation order differences).

**Files:**
- Modify: `tests/test_assign_dslab.py` (append a parametric test)

- [ ] **Step 1: Append the test**

Append to `tests/test_assign_dslab.py`:

```python
def _run_compare_script(D: int, K: int, dtype: str = "float16") -> tuple[int, str]:
    script = f"""
import os
import torch
from flash_kmeans_cuda import _C

torch.manual_seed(0)
B, N, D, K = 1, 2048, {D}, {K}
dtype = torch.{dtype}
x = torch.randn(B, N, D, device='cuda', dtype=dtype)
centroids = x[:, :K].contiguous()
x_sq = (x.float() ** 2).sum(-1).contiguous()
c_sq = (centroids.float() ** 2).sum(-1).contiguous()

# Run dslab.
os.environ['FKC_DSLAB'] = '1'
ids_dslab = _C.euclid_assign(x, centroids, x_sq, c_sq, None).clone()

# Run narrow (legacy).
os.environ.pop('FKC_DSLAB', None)
os.environ['FKC_NARROW'] = '1'
ids_narrow = _C.euclid_assign(x, centroids, x_sq, c_sq, None).clone()

disagree = (ids_dslab != ids_narrow).float().mean().item()
assert disagree < 0.03, (
    f"D={{D}} K={{K}} dtype={{dtype}}: dslab vs narrow disagree {{disagree:.3%}} "
    f"(threshold 3% — tied-distance + cross-accumulation order)"
)
print(f'OK D={{D}} K={{K}} disagree={{disagree:.4%}}')
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120,
    )
    return proc.returncode, proc.stdout + proc.stderr


@pytest.mark.parametrize("D", [192, 224, 256, 320, 384])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_dslab_matches_narrow(D, dtype):
    """D-slab kernel agrees with legacy narrowk32 on the same shape (within 3%)."""
    rc, out = _run_compare_script(D=D, K=512, dtype=dtype)
    assert rc == 0, f"D={D} dslab vs narrow failed:\n{out}"
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_assign_dslab.py -v`
Expected: 12 cases pass (2 dtypes × 5 D values for the new test, plus the 2 smoke cases).

If any D fails with disagreement > 3%, STOP and report as BLOCKED. Common causes:
- Off-by-one in `slab_off += slab_w` for non-uniform partitions (D=192/224/320 are most likely to surface this).
- Incorrect `kNumSlabs` computation for partitions with internal zeros.

- [ ] **Step 3: Commit**

```bash
git add tests/test_assign_dslab.py
git commit -m "Test dslab agrees with narrowk32 across all 5 target D values"
```

---

## Task 8: Promote D-slab variants in `kD192`-`kD384` policy rows

With correctness proven by Tasks 4–7, route the autotuner to prefer D-slab over the legacy variants for D ∈ {192, 224, 256, 320, 384}.

**Files:**
- Modify: `flash_kmeans_cuda/csrc/assign/assign_policy.cu`

- [ ] **Step 1: Replace the five rows**

Locate the existing `constexpr PolicyRow kD192 = { ... };` definition. Replace `kD192`, `kD224`, `kD256`, `kD320`, `kD384` with these new orders (keep `kD64`, `kD96`, `kD128`, `kD320`-companion-N1/N4 untouched):

```cpp
constexpr PolicyRow kD192 = {
  &V_DSLAB_W8_N2_D192, &V_DSLAB_W8_D192,
  &V_WIDEK96_W8_N2_D192, &V_WIDEK96_W8_D192,
  &V_NARROWK32_W4_N2_D192, &V_NARROWK32_W4_N2,
  &V_NARROWK32_W4, &V_NARROW_4,
};

constexpr PolicyRow kD224 = {
  &V_DSLAB_W8_N2_D224, &V_DSLAB_W8_D224,
  &V_WIDEK96_W8_N2_D224, &V_WIDEK96_W8_D224,
  &V_NARROWK32_W4_N2_D224, &V_NARROWK32_W4_N2,
  &V_NARROWK32_W4, &V_NARROW_4,
};

constexpr PolicyRow kD256 = {
  &V_DSLAB_W8_N2_D256, &V_DSLAB_W8_D256,
  &V_WIDE_3_W8_N2_D256, &V_WIDE_3_W8_D256,
  &V_NARROWK32_W4_N2_D256, &V_NARROWK32_W4_N2,
  &V_NARROWK32_W4, &V_WIDE_3_W4,
};

constexpr PolicyRow kD320 = {
  &V_DSLAB_W8_N2_D320, &V_DSLAB_W8_D320,
  &V_NARROWK32_W4_N2_D320, &V_NARROWK32_W4_N2,
  &V_NARROWK32_W4, &V_WIDE_3_W4, &V_NARROW_4, &V_DEEP_2_W4,
};

constexpr PolicyRow kD384 = {
  &V_DSLAB_W8_N2_D384, &V_DSLAB_W8_D384,
  &V_NARROWK32_W4_N2_D384, &V_NARROWK32_W4_N2,
  &V_NARROWK32_W4, &V_WIDE_3_W4, &V_NARROW_4, &V_DEEP_2_W4,
};
```

`kRowsN1` / `kRowsN4` (the per-N_TILES tables) keep their existing rows for these D values — D-slab is only NT∈{1,2}; FKC_NTILES=4 still routes to the legacy `narrowk32_w4_n4` path. This is a deliberate omission; the autotuner picks the best N_TILES via the default `kRowsN2` path.

- [ ] **Step 2: Build and run all the equivalence tests**

Run:
```bash
uv run python setup.py build_ext --inplace && \
uv run pytest tests/test_assign_dispatch_equiv.py tests/test_correctness.py \
              tests/test_persistent.py tests/test_assign_autotune.py \
              tests/test_assign_dslab.py -x -q
```
Expected: all pass. `test_all_locked_d_values_dispatch` exercises every D ∈ {64, 96, 128, 192, 224, 256, 320, 384} and now routes D ∈ {192..384} through D-slab — must produce correct cluster_ids within the per-dtype thresholds (2% fp16 / 5% bf16).

If any D fails, STOP and report as BLOCKED. Most likely suspect: D-slab kernel produces > 2% disagreement at K=256 (test parameter) where mma rounding cascades from cross-slab accumulation become visible. Mitigation: investigate; do NOT silently widen the threshold.

- [ ] **Step 3: Commit**

```bash
git add flash_kmeans_cuda/csrc/assign/assign_policy.cu
git commit -m "Promote D-slab variants ahead of legacy in D in {192..384} rows"
```

---

## Task 9: Bench acceptance — D=192/256 mega ≥ 180 TFLOPS, D=128 within ±2%

**Files:** none modified; this is a verification task.

- [ ] **Step 1: Benchmark D=128 (regression guard)**

Run:
```bash
uv run python benchmarks/bench_d_sweep.py --d 128 --n 32768 --k 8192 --rounds 30 --warmup 5
```
Capture the TFLOPS number. Compare to the pre-D-slab baseline of ~202 TFLOPS. **Must be within ±2% (i.e., 198-206 TFLOPS).** If outside, STOP and report as BLOCKED — the D-slab work was not supposed to touch the D=128 path.

- [ ] **Step 2: Benchmark D=192/256 mega-K (primary acceptance)**

Run:
```bash
uv run python benchmarks/bench_d_sweep.py --d 192 256 --n 32768 --k 8192 --rounds 30 --warmup 5
```
Both D values must report **≥ 180 TFLOPS**. If either falls short (e.g., D=192 lands at 160), the D-slab kernel underperforms vs projection.

If acceptance fails, the most actionable diagnostic is to enable verbose autotune and check which variant was picked:
```bash
FKC_AUTOTUNE_VERBOSE=1 uv run python benchmarks/bench_d_sweep.py --d 192 --n 32768 --k 8192 --rounds 1 --warmup 0
```
Expected output: `probe[0]=dslab_w8_n2_d192 -> 0.4xx ms` (or similar) with dslab being the winner. If narrowk32 is still winning, the dslab kernel is genuinely slower — STOP and report.

- [ ] **Step 3: Benchmark the rest (soft acceptance)**

Run:
```bash
uv run python benchmarks/bench_d_sweep.py --d 224 320 384 --n 32768 --k 8192 --rounds 30 --warmup 5
```
Each D must be **≥ pre-D-slab baseline**:
- D=224 ≥ 149 TFLOPS
- D=320 ≥ 167 TFLOPS
- D=384 ≥ 161 TFLOPS

If a D regresses, the autotuner correctly fell back to legacy narrowk32 (since narrowk32 was already the de facto baseline). The regression would mean dslab was wrongly chosen; investigate via the verbose autotune log.

- [ ] **Step 4: Small-N regression check**

Run:
```bash
uv run python benchmarks/bench_d_sweep.py --d 192 256 --n 2048 --k 8192 --rounds 20 --warmup 3
```
D=192 / D=256 dslab must beat the pre-D-slab narrowk32 baseline by **≥ 10%** at small N. If not, the per-CTA prologue overhead is dominating; flag for a follow-up split-K spec but don't block the merge.

- [ ] **Step 5: Final full test sweep**

Run: `uv run pytest tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit (if any tuning needed)**

If the acceptance benchmarks pass, no commit needed — the bench is verification, not a code change. If the autotuner needed reordering (e.g., dslab is faster than expected and should drop the legacy fallbacks), make the row reordering and commit:

```bash
git add flash_kmeans_cuda/csrc/assign/assign_policy.cu
git commit -m "Reorder kD<X> row based on bench results"
```

---

## Self-Review Notes

- **Spec coverage:** Every locked decision in the spec maps to a task: `D_SLAB granularity = 128 max` → Task 1 catalog entries with explicit `S0..S3` partition. Per-D partitions (192→[128,64], etc.) → Task 6 catalog. Tile config BN=128 BK=128 STAGES=2 8w → Task 1 factory + Task 6 entries. SMEM_PAD_SLAB=0 → Task 1 default + Task 5 SMEM allocation. 20 new instantiations (10 variants × 2 dtypes) → Tasks 3 + 6. Dispatcher integration → Task 8. Spec acceptance criteria → Task 9.
- **Placeholder scan:** Two soft references — Task 2's "rest of the body unchanged" and Task 5's "copy the legacy 4-fp32-reg dist update". Both reference specific code blocks in `assign_sm80_kernel.cuh` that the implementer can reach via `git show :assign_sm80_kernel.cuh`. Not a placeholder in the planning sense (no missing decision); just deferring 200 lines of literal kernel code that wouldn't fit in a plan markdown anyway.
- **Type consistency:** `LaunchCtx`, `Variant`, `PolicyRow`, `VariantView`, `EnvKnobs`, `MAX_CAND` are reused from the existing infrastructure. `MAX_SLABS = 4` is defined once in `assign_dslab_variants.h` (Task 1) and referenced by name in Tasks 2 and 5. Partition uses `S0..S3` consistently (4 explicit ints, no `std::array` non-type template parameter).
- **Risk acknowledged:** Task 5 step 5 explicitly tells the implementer to STOP rather than widen tolerance; Task 9 step 2 explicitly tells the implementer to STOP if D=128 regresses (not silently accept).
