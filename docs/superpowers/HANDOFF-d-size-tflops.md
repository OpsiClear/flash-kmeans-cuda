# Handoff: Pushing assign-kernel TFLOPS past the catalog wall

**Branch:** `optimize/d-size-tflops-20260504`
**Last commit:** `0cbf6f8` — "exp6: keep SMEM_PAD plumbing, drop dead PAD=0 catalog (bank conflicts 3-6x slower)"
**Repo:** `C:\Users\HEQ\Projects\flash-kmeans-cuda`
**GPU:** RTX 4090 (Ada / sm_89), 99 KB SMEM-per-block-optin cap, ~330 TFLOPS fp16 tensor-core theoretical

---

## Task & target

Optimize the **assign kernel** (Euclidean argmin over centroids) at K=8192 mega-K shapes for D ∈ {128, 192, 224, 256, 320, 384}, fp16, B=1, N=32768. Single composite metric: **sum of TFLOPS across the 6 D values**, higher = better.

**Current state (sustained-bench, slightly thermal-throttled):**

| D | TFLOPS | Winner kernel | % of D=128 roofline |
|---|---|---|---|
| 128 | 200 | `widek96_w8_n2_d128` | 100% |
| 192 | 195 | `widek48_w8_n2_d192` | 98% |
| 224 | 188 | `widek32_w8_n2_d224` | 94% |
| **256** | **169** | `narrowk48_w4_n4_d256` | **84% — gap** |
| **320** | **162** | `narrowk32_w4_n2_d320` | **81% — gap** |
| **384** | **174** | `narrowk32_w4_n4_d384` | **87% — gap** |

**Target:** push D ∈ {256, 320, 384} from ~85% to ≥95% of the D=128 roofline (i.e., ≥190 TFLOPS each). D=128/192/224 are already maxed within the current kernel template.

---

## Why D ≥ 256 is stuck (the wall)

The kernel uses mma.m16n8k16 with operand layout that forces:
- BLOCK_N (BN) ∈ {64, 128} with WARP_M = BN / WARPS ≥ 16 (multiple of 16). So WARPS ≤ 8 at BN=128, ≤ 4 at BN=64.
- BLOCK_K (BK) must be a multiple of 16 (ldmatrix.x4 covers 2 N-atoms; static_assert).
- BLOCK_D = 16 hard-coded (mma K-step granularity).
- SMEM footprint: `BN*(D+SMEM_PAD)*2 + STAGES*BK*(D+SMEM_PAD)*2 + STAGES*BK*4` ≤ 99 KB.

At D=256, BN=128 BK=32 STAGES=2 SMEM_PAD=8 = 102 KB — over by 3 KB.
- SMEM_PAD=0 fits but causes catastrophic bank conflicts (D%32==0 → all rows hit same bank). Empirically 3-6× slower.
- BK=48/64 don't fit. BN=64 fits but halves arithmetic intensity per CTA (WIDEK family loses to NARROW family there).

D=320 / D=384 are even tighter. Only BN=64 BK=32 STAGES=2 fits.

13 catalog experiments confirmed empirically that no kernel-template-preserving change unlocks more perf at D ≥ 256.

## Why D=192/224 won (template within reach)

- BN=128 W=8 (BK=32 / BK=48) doubles per-CTA arithmetic intensity vs BN=64 W=4. Fits SMEM at D ≤ 224 (BK=32) and D ≤ 192 (BK=48).
- NT=4 persistence helps amortize launch overhead at N=32768.
- Wider-K narrow probes (BK=48, BK=64 at BN=64) are also competitive.

---

## Proven dead-ends (do NOT re-try)

`results.tsv` has full history. Key losing experiments and reasons:

| Idea | Loss | Reason |
|---|---|---|
| WIDEK16 W8 D=256 (BK=16 fits PAD=8) | -0.3 (noise) | BK=16 too small; K-chunk count 512 |
| WIDEK32 STAGES=3 D=192 | -14 | extra stage doesn't help when BK=48 STAGES=2 already wins |
| NT=8 WIDEK | -79 | undersaturates 84-SM 4090 (32 CTAs not enough) |
| WIDEK W=4 (longer mma chains) | -6 | half warps reduce per-cycle issue rate |
| NT=1 WIDEK | -40 | too many CTAs, launch overhead dominates |
| NARROWK64 STAGES=1 D=256 | -7 | bigger BK doesn't compensate for losing pipelining |
| `async_csq=false` | -198 | sync c_sq store path is K-chunk-serialized at K=mega |
| BN=32 W=2 NT=4 tiny-CTA | -148 | 2 warps/CTA = catastrophic occupancy at SMEM cap |
| `#pragma unroll 4` (was 8) | -72 | lighter unroll defeats software pipelining of ldmatrix vs mma |
| SMEM_PAD=0 PAD-template-param at D=224/256 | unusable | bank conflicts when D % 32 == 0 |

---

## Real next moves (in order of viability after validation)

### Move 1 — D-slab kernel inter-slab pipelining (1-2 weeks, projected +20-30% at D ≥ 192) — **highest priority**

The dslab kernel scaffold (`flash_kmeans_cuda/csrc/assign/assign_sm80_dslab_kernel.cuh`) is already in place; the perf bug is `cp_async_wait_all()` between slabs killing overlap. Fix:

- Allocate **2 SMEM stages** for both x_slab and c_slab (currently 1 each).
- Replace `wait_all` with `cp_async_wait_group<1>` so slab N+1's load overlaps slab N's mma.
- Drop BN to 64 OR BK to 64 to fit the doubled SMEM under 99 KB.

At D=384 with BN=64 BK=64 STAGES=2 D-slab: ~97 KB, projected ≥190 TFLOPS. Spec at `docs/superpowers/specs/2026-05-03-assign-d-slab-tiling-design.md`; outcome notes at `docs/superpowers/specs/2026-05-03-assign-d-slab-tiling-results.md`.

This is the single highest-leverage lever for D ∈ {192, 256, 320, 384} since it sidesteps the SMEM-cap wall entirely.

### Move 2 — `mma.m16n8k32` (~1-2 days, projected +0-5%) — **lower-priority**

Already analyzed in this session. Switching to m16n8k32 would halve the d_step iteration count but Ada's m16n8k32 throughput is 1 inst/4 clk vs 2× m16n8k16 = 2 inst × 2 clk = same cycle count. Doubling ldmatrix per d_step also cancels the inner-loop reduction. Win comes only from issue-rate slack and reduced static instruction count.

**Files to touch:**
- `flash_kmeans_cuda/csrc/common/ptx.cuh` — add `mma_m16n8k32_fp16_acc_fp16` helper. PTX: `mma.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16` with 8 A regs, 4 B regs, 2 D regs per thread.
- `flash_kmeans_cuda/csrc/assign/assign_sm80_kernel.cuh` (line 318 d_step loop) — add a `int K_STEP=16` template parameter. When K_STEP=32, double the A/B reg loads and call the new mma helper.
- Plumb K_STEP through `launch_typed` / `make_variant<>`.
- Add K_STEP=32 catalog entries.

**Risk:** ldmatrix.x4 lane-to-source mapping changes when loading 32 K cols at once. Wrong layout = silent correctness break (caught by `test_assign_dispatch_equiv.py::test_all_locked_d_values_dispatch`).

Skip unless Move 1 underdelivers.

### Move 2 — D-slab kernel inter-slab pipelining (1-2 weeks, projected +20-30% at D ≥ 192)

The D-slab kernel `flash_kmeans_cuda/csrc/assign/assign_sm80_dslab_kernel.cuh` is already scaffolded (Tasks 1-9 of the prior session). It correctly produces cluster_ids but is 2-3× SLOWER than legacy because the per-slab `cp_async_wait_all` kills overlap.

The fix:
1. Use 2 SMEM stages on x_slab and c_slab (currently 1 each).
2. Replace `cp_async_wait_all()` with `cp_async_wait_group<1>()` so slab N+1's load overlaps slab N's mma.
3. Reduce BN to 64 OR BK to 64 to fit the doubled SMEM under 99 KB.
4. Target SMEM at D=384: 2×16 KB x_slab + 2×32 KB c_slab + ε = ~97 KB ✓.

**Spec:** `docs/superpowers/specs/2026-05-03-assign-d-slab-tiling-design.md`
**Plan:** `docs/superpowers/plans/2026-05-03-assign-d-slab-tiling.md`
**Outcome doc:** `docs/superpowers/specs/2026-05-03-assign-d-slab-tiling-results.md` (explains why current dslab is slow + the fix).

This is the single highest-leverage lever for D ∈ {192, 256, 320, 384}.

### Move 3 — split-K reduction (1 week, helps small-N + mega-K)

Add a 2-pass kernel: pass 1 emits `(best_dist, best_idx)` per (BN_rows × BK_partial) tile to scratch; pass 2 reduces. Frees SMEM (each CTA owns less K) and improves occupancy at small N. Compounds with Move 2.

Skip until Moves 1 and 2 measure their actual headroom.

---

## Benchmark method (reproducible)

Single composite metric script: `scripts/windows/autotune_metric.py`

```bash
DISTUTILS_USE_SDK=1 uv run --no-sync python scripts/windows/autotune_metric.py
# emits: TOTAL_TFLOPS=<float>
# stderr also prints: Per-D TFLOPS: D128=..., D192=..., ...
```

Internally runs `benchmarks/bench_d_sweep.py --n 32768 --k 8192 --d 128 192 224 256 320 384 --rounds 50 --warmup 10 --outer 3`.

**Reproducibility caveats:**
- GPU thermal throttling causes ~5-15% run-to-run noise on sustained loops. Same shape can read 175 vs 200 TFLOPS depending on prior workload.
- Best signal: run multiple times back-to-back, take median, or use the verbose autotune (cold-cache, 3-iteration min) which is much more stable per-cell. Verbose-mode peak numbers are the row in the table at the top of this doc.

**Verbose autotune (per-cell winners + best-of-3 timing):**

```bash
FKC_AUTOTUNE_VERBOSE=1 uv run --no-sync python benchmarks/bench_d_sweep.py \
    --d 128 192 224 256 320 384 --n 32768 --k 8192 --rounds 1 --warmup 0 \
    2>&1 | Select-String "fkc autotune.* probe\[0\]|^D|^[0-9]"
```

Each `probe[0]` line is the autotuner's winning candidate per `(dtype, D_idx, k_bucket)` cell with its min-of-3 timing.

---

## Build steps (Windows / MSVC + CUDA 13.2)

The Windows toolchain has a Git-link.exe-shadowing-MSVC-link.exe gotcha. Use the wrapper:

```cmd
scripts\windows\autotune_step.bat
```

This wrapper:
1. Calls `scripts\windows\_setup_env.bat` to set up `vcvars64`, strip Git's bundled `link.exe` from PATH, set `DISTUTILS_USE_SDK=1`.
2. Sets `TORCH_CUDA_ARCH_LIST=8.9` and `MAX_JOBS=1` (avoids nvcc OOM on cl.exe + ptxas memory pressure).
3. Runs `uv run --no-sync python setup.py build_ext --inplace` (incremental).
4. Runs the metric script and emits TOTAL_TFLOPS.

**Manually:**

```cmd
scripts\windows\_setup_env.bat
set TORCH_CUDA_ARCH_LIST=8.9
set MAX_JOBS=1
uv run --no-sync python setup.py build_ext --inplace
```

For broad-arch release builds, override `TORCH_CUDA_ARCH_LIST` to include sm_80/86/89/90/100/120 (slower; not needed for the tuning loop).

**Other useful scripts:**
- `scripts\windows\run_exp.bat` — full clean install + smoke tests + benchmark
- `scripts\windows\run_exp_t.bat` — same but compares vs Triton oracle
- `scripts\windows\profile_kernel.bat` — Nsight Compute kernel profile

---

## Auto-tune loop infrastructure (already in place)

If continuing in the same loop style:
- `results.tsv` (gitignored) — append-only TSV of every experiment's metric.
- `scripts/windows/autotune_metric.py` — emits single `TOTAL_TFLOPS=<n>` line.
- `scripts/windows/autotune_step.bat` — build + bench in one call.

To run the loop on a new branch:

```bash
git checkout -b optimize/<new-tag> 0cbf6f8
printf "commit\tmetric\tresource\tstatus\tdescription\n" > results.tsv
# Run baseline first to anchor, then start experimenting.
```

---

## Key files for the next session

| File | Role |
|---|---|
| `flash_kmeans_cuda/csrc/assign/assign_policy.cu` | Variant catalog + per-D policy rows. Most catalog tuning happens here. |
| `flash_kmeans_cuda/csrc/assign/assign_sm80_kernel.cuh` | The mma kernel template. Edits here = real perf wins. mma instruction at line 52 (`mma_m16n8k16_fp16_acc_fp16`); d_step inner loop at line 317. |
| `flash_kmeans_cuda/csrc/assign/assign_kernel_launch.h` | `launch_typed` / `launch_typed_select_csq` template plumbing. SMEM_PAD already a template param. |
| `flash_kmeans_cuda/csrc/assign/assign_variants.h` | `VariantSpec` / `make_variant`. Add new template params here when extending. |
| `flash_kmeans_cuda/csrc/common/ptx.cuh` | mma + cp.async + ldmatrix PTX wrappers. New instructions get added here. |
| `flash_kmeans_cuda/csrc/assign/assign_sm80_dslab_kernel.cuh` | D-slab kernel (Move 2 starting point). |
| `tests/test_assign_dispatch_equiv.py` | Correctness regression suite. Run after every kernel change. |
| `tests/test_assign_dslab.py` | D-slab smoke + cross-kernel agreement. |

## Existing infrastructure highlights

- **SMEM_PAD is already a template parameter** on the kernel (added in exp5). Default is 8. Pass other values via the `Pad` template arg in `make_variant<...>`. Just don't pass 0 at D % 32 == 0 (bank conflict suicide).
- **In-memory autotune cache** keyed by `(dtype, D_idx, K_bucket)` is in `assign_autotune.cu`. Probes top-3 candidates per cell; serves cached winner thereafter. Lock-free hot path. `FKC_AUTOTUNE=0` disables; `FKC_AUTOTUNE_VERBOSE=1` logs probes.
- **Force-route knobs**: `FKC_W4=1`, `FKC_WIDE3=1`, `FKC_NARROW=1`, `FKC_NTILES={1,2,4}`, `FKC_DSLAB=1`, `FKC_ASSIGN_DEEP_TILE=1` — bypass autotune for specific developer experiments.
- **Per-call env reads**: env knobs are read on every launch, so they take effect immediately (no process restart needed).

## Recommended first moves for the new session

1. Read `results.tsv` and this doc.
2. Run `scripts\windows\autotune_step.bat` to confirm baseline reproduces (~1043 TFLOPS).
3. Run the verbose autotune (`FKC_AUTOTUNE_VERBOSE=1` ...) to see per-cell winners.
4. Pick one of Move 1 / 2 / 3. **Move 1 (mma.m16n8k32) is the lowest-risk, smallest-rewrite path.**
5. Branch with `optimize/m16n8k32` or similar. Reset `results.tsv`. Bench against baseline 1043.
