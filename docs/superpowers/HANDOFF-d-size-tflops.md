# Handoff: Pushing assign-kernel TFLOPS past the catalog wall

**Branch:** `optimize/dslab-pipeline-20260505`
**Last commit:** `a3fa274` — "Update handoff: validate asymmetric-pad as dead end; reorder priorities"
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

## Small-D addendum (D=1..16)

The assign path now has explicit arbitrary-small-D coverage for the same
mega-K shape (`B=1, N=32768, K=8192`, fp16). D1 fp16/bf16 uses an exact
sorted-centroid path (`sort K`, binary-search each point, compare neighbors,
duplicate-run tie handling). D2 routes through `launch_assign_safe`'s
shared-centroid tiled fixed-D scalar kernel (`BN=128, BK=512`). D3 routes to
a padded-D16 SM80 tensor-core variant with fp32 accumulation; fp16
accumulation was faster but lost too many near-neighbor ties at low D.
`D=4..15` route to padded-D16 SM80 tensor-core variants by default, and
`D=16` uses the normal D16 SM80 row.

Final full small-D sweep (`bench_d_sweep.py --n 32768 --k 8192 --d 1 ... 16
--rounds 20 --warmup 5 --outer 3`):

| D | ms | TFLOPS | Default route |
|---:|---:|---:|---|
| 1 | 0.0597 | 8.99 | sorted-centroid exact D1 |
| 2 | 0.1097 | 9.79 | tiled scalar, `BN=128`, `BK=512` |
| 3 | 0.1475 | 10.92 | padded D16 tensor-core, fp32 acc, `BK=128` |
| 4 | 0.1088 | 19.74 | padded D16 tensor-core, `BK=256` |
| 5 | 0.1093 | 24.56 | padded D16 tensor-core, `BK=256` |
| 6 | 0.1096 | 29.39 | padded D16 tensor-core, `BK=256` |
| 7 | 0.1092 | 34.41 | padded D16 tensor-core, `BK=256` |
| 8 | 0.0869 | 49.42 | padded D16 tensor-core, `BK=256` |
| 9 | 0.1106 | 43.69 | padded D16 tensor-core, `BK=256` |
| 10 | 0.1113 | 48.23 | padded D16 tensor-core, `BK=256` |
| 11 | 0.1105 | 53.45 | padded D16 tensor-core, `BK=256` |
| 12 | 0.1110 | 58.06 | padded D16 tensor-core, `BK=256` |
| 13 | 0.1374 | 50.81 | padded D16 tensor-core, `BK=256` |
| 14 | 0.1114 | 67.46 | padded D16 tensor-core, `BK=256` |
| 15 | 0.1109 | 72.62 | padded D16 tensor-core, `BK=256` |
| 16 | 0.0888 | 96.74 | D16 tensor-core, `BK=256` |

Focused repeat checks are less noisy than the full sweep for the new D3 row:
`D=3` isolated at the same shape measured 0.1108 ms / 14.53 TFLOPS with
`pad16_widek128_w8_d3_f32acc`.

Correctness:
- Smoke at `N=4096, K=2048, D=1..4`: D1 and D2 had 0.0000% disagreement vs
  the fp32 reference, D3 fp32-accum padded TC had 0.0244%, and D4 had 0.6592%.
- Target-size representative check at `N=32768, K=8192`: D2 showed 0.0061%
  disagreement in one run, and D3 fp32-accum padded TC was exact in that run.
- Previous full smoke at `N=4096, K=2048, D=1..16`: padded tensor-core
  dimensions D4-D16 were all below 0.66% disagreement.
- Inline fp16/bf16 sanity at `N=1024, K=128` passed representative
  D1/D2/D3/D4/D5/D7/D8/D15/D16 cases.

Do not route D1-D2 through the padded tensor-core path by default. Padded TC
with fp16 accumulation measured ~0.106-0.110 ms, but disagreement was
unacceptable at low D: D1 78.49%, D2 10.67% in smoke, and D3 also lost too
many ties. D2 padded TC with fp32 accumulation fixed correctness but slowed to
~0.192 ms, so D2 stays on the tiled scalar path. D3 uses padded TC only with
fp32 accumulation; `BK=128` beat the original `BK=256` fp32-accum candidate.

Small-D probes rejected after the keep:
- D3 `BLOCK_N=256` regressed to 0.443 ms; `BLOCK_N=32` regressed to 0.430 ms. `BLOCK_N=64` was best.
- D3 direct squared-distance form removed the `c_sq` load but was slower (0.404 ms) than raw-distance `c_sq - 2*dot`.
- D8/D16 BK512 and BK384 both lost to BK256 in autotune probes; BK256 `N_TILES=2` also lost to BK256 `N_TILES=1`.
- Rows-per-thread scalar reuse lost for `D=1..7`: mapping D1-D4 to 4 rows/thread
  and D5-D7 to 2 rows/thread regressed D1/D3/D7 to 0.562/0.721/0.978 ms vs
  0.255/0.385/0.568 ms. Keep one point per thread; CTA count matters more
  than per-thread centroid reuse at the target N.
- Shared-centroid tiled scalar is kept for D2. It loads `BK=512` centroids
  and `c_sq` values into CTA shared memory once, then 128 point threads reuse
  the tile. BK1024 had one isolated 0.1052 ms run but did not reproduce
  cleanly in later all-D/all-N sweeps, so the cleaner BK512 baseline is kept.
  BK2048 regressed to 0.1254 ms. BN256 lost; a D3-only BN64 specialization did
  not reproduce as a sustained win in the final binary.
- D3 padded TC rejected variants: fp16-accum padded TC was fast but inaccurate;
  fp32-accum `BK=256` was correct but slower (~0.157 ms probe); `BK=128_N2`
  regressed (~0.172 ms probe). These slower fallback variants are no longer
  compiled; keep only `pad16_widek128_w8_d3_f32acc`.
- Sorted-centroid D1 is kept for fp16/bf16. It improved D1 from the tiled
  scalar 0.105 ms to 0.082 ms in the full sweep (focused runs reached
  ~0.062 ms). The kernel scans duplicate-value runs around the two binary
  search neighbors so fp16 duplicate centroids return the same lowest original
  centroid id as the fp32 reference.
- Padded tensor-core routing is kept for D4-D15. It moves D4-D7 from the
  tiled-scalar 0.15-0.21 ms range to ~0.11-0.14 ms while staying below the
  existing fp16 disagreement tolerance at target K.

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
| `#pragma unroll 16` (was 8) | crash/discard | Opposite-direction D-loop unroll probe did not produce a usable binary. `ptxas` ran for several minutes, the build wrapper returned nonzero with no compiler diagnostic beyond setup headers, and `_C.cp312-win_amd64.pyd` timestamp did not update. Treat as code-size/compiler-resource blowup unless proven otherwise. |
| SMEM_PAD=0 PAD-template-param at D=224/256 | unusable | bank conflicts when D % 32 == 0 |
| SMEM_PAD=4 / SMEM_PAD=16 probes | discard | PAD=4 is structurally invalid in the current layout: odd rows put `cp.async`/`ldmatrix` destinations on 8-byte offsets and the D=256 wide-tile probe hit CUDA misaligned-address. PAD=16 preserves alignment but was slower/noisy on the current narrow winners: D256 pad16 0.928-1.367 ms vs pad8 0.819, D320 pad16 ~1.286 ms vs pad8 ~1.298 in probe but worse in the measured row. |
| D-slab inter-slab pipelining, D=256 only | discard | Correct but still much slower: BN=128/BK=64/W8/S2 probed 3.137 ms, BN=64/BK=128/W4/S2 probed 2.312 ms, current `narrowk48_w4_n4_d256` probed 0.812 ms at N=32768/K=8192/D=256. Double-buffering x_slab+c_slab does not fix the core cost that x_slab is reloaded for every K chunk. |
| Autotuner `kProbeTopN=4` | discard | Fourth candidates did not beat selected winners in verbose mega sweep: D128 `wide_3_w8` 0.422 ms vs 0.343, D192 `widek32_w8_n4_d192` 1.042 vs 0.497, D224 `narrowk64_w4_n4_d224` 0.695 vs 0.584, D256 `narrowk32_w4_n2_d256` 0.889 vs 0.811, D320 `narrowk32_w4` 1.448 vs 1.064, D384 `narrowk32_w4` 1.834 vs 1.133. |
| `N_TILES=3` large-D catalog probes | discard | Middle point between NT=2 and NT=4 was worse in every targeted mega-K row: D192 `widek48_w8_n3_d192` 0.932 ms vs 0.495, D224 `widek32_w8_n3_d224` 0.863 vs 0.581, D256 `narrowk48_w4_n3_d256` 1.204 vs 0.811, D320 `narrowk32_w4_n3_d320` 1.573 vs ~1.06, D384 `narrowk32_w4_n3_d384` 1.749 vs ~1.18. |
| `N_TILES=5/6` BN=64 large-D probes | discard | Tried the middle between NT=4 and NT=8 for D=256/320/384 so CTA count stays near the 84-SM 4090 target (NT=5 -> 103 CTAs, NT=6 -> 86 CTAs at N=32768). Still slower: D256 N5/N6 1.012/1.206 ms vs N4 0.819, D320 N5 1.320 vs N2/N4 ~1.07, D384 N5 1.467 vs N2/N4 ~1.19. |
| BK=16 S3/S4 for D=320/384 | discard | BK=32 S3/S4 does not fit SMEM at these D values, so BK=16 was tested with deeper cp.async staging. Doubling K chunks was not recovered by extra stages: D320 BK16 S3/S4 1.355/1.398 ms vs BK32 1.063, D384 BK16 S3/S4 1.480/1.469 vs BK32 1.187. |
| Full-N specialization | discard | Added a `FullN` template flag to skip N-tail checks when `N % (BN*NT) == 0` and probed current target-D winners. Cold probes showed small wins for some rows (D192 0.480 vs 0.492, D256 0.775 vs 0.790, D320 NT4 full-N 0.994 vs ~1.016, D384 1.124 vs 1.143), but sustained target sweeps did not improve and sometimes regressed badly. Complexity is not justified by noisy sub-2% probe wins. |
| Two-pass split-K / partial-K prototype | discard | Implemented behind `FKC_SPLITK=1`: pass 1 reused the SM80 MMA kernel over K partitions and emitted `(best_dist,best_idx)` scratch, pass 2 reduced scratch. Correct after fixing the partition-tail validity check (`k_start_local + k_in_chunk < k_total`, not `k_global < K`). Target full sweep still lost: split-K total 884.8 TFLOPS vs no-split 890.4 under the same run (`rounds=20,warmup=5,outer=3`). Small N=8192 smoke was faster, but target N=32768 already saturates enough that scratch/reducer overhead erases the gain. |
| `cp.async.ca` instead of `cp.async.cg` | crash/discard | Tried switching the common async-copy primitive to cache at all levels. Build returned nonzero before producing a fresh `assign_sm80.obj` or updating `_C.cp312-win_amd64.pyd`; no usable benchmark result. The restored `.pyd` is the clean pre-probe binary. |
| Dense `mma.m16n8k32` fp16 | invalid | Do not implement the earlier proposed dense path. NVIDIA PTX ISA 9.2 lists dense `.f16` MMA for `m16n8k16`; `.f16 m16n8k32` is listed for sparse/ordered-metadata MMA, not dense `mma.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16`. |

---

## Real next moves (after 2026-05-05 validation)

The previous first-move recommendations are now closed:

- Dense `mma.m16n8k32` is not a valid dense fp16 MMA instruction form.
- D-slab inter-slab pipelining was correct but much slower on D=256, and the reason appears structural: x slabs are still reloaded across K chunks.
- Small catalog/template probes (`kProbeTopN=4`, `N_TILES=3`, `N_TILES=5/6`, BK=16 S3/S4, `SMEM_PAD=4/16`, full-N specialization) did not uncover a hidden winner.
- The straightforward two-pass split-K prototype was correct, but did not improve the target N=32768 metric.

### Move 1 — integrated K-partitioning without scratch (research)

The discarded split-K result says the idea only has room if the extra pass and scratch traffic disappear. A future attempt would need to keep partial winners on device without materializing `(B, parts, N)` scratch, for example via cooperative groups, persistent CTAs, or a fused reducer design.

Do not re-add the simple two-pass scratch prototype unless the benchmark target changes to smaller N; it was faster in a small N=8192 smoke but lost at the project target N=32768.

### Move 2 — robustness / measurement cleanup

These are useful for handoff quality but are not expected to raise the best TFLOPS:

- Rebuild after restoring any discarded source experiment so the local `.pyd` is not stale.
- Consider `kProbeIters=5` only to reduce winner-selection noise; do not count it as a throughput optimization unless the canonical `autotune_step.bat` improves.
- Keep using verbose sweeps to reject candidates before spending a full 5-minute metric run.

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
- `scripts\windows\run_exp_t.bat` — build + smoke tests + assign-only Triton comparison
- `scripts\windows\profile_kernel.bat` — Nsight Compute kernel profile

---

## Auto-tune loop infrastructure (already in place)

If continuing in the same loop style:
- `results.tsv` (gitignored) — append-only TSV of every experiment's metric.
- `scripts/windows/autotune_metric.py` — emits single `TOTAL_TFLOPS=<n>` line.
- `scripts/windows/autotune_step.bat` — build + bench in one call.

To run the loop on a new branch:

```bash
git checkout -b optimize/<new-tag>
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
| `flash_kmeans_cuda/csrc/assign/assign_sm80_dslab_kernel.cuh` | D-slab scaffold retained for reference; current pipelined D=256 probe was slower than legacy. |
| `tests/test_assign_dispatch_equiv.py` | Correctness regression suite. Run after every kernel change. |
| `tests/test_assign_dslab.py` | D-slab smoke + cross-kernel agreement. |

## Existing infrastructure highlights

- **SMEM_PAD is already a template parameter** on the kernel (added in exp5). Default is 8. Pass other values via the `Pad` template arg in `make_variant<...>`. Just don't pass 0 at D % 32 == 0 (bank conflict suicide).
- **In-memory autotune cache** keyed by `(dtype, D_idx, K_bucket)` is in `assign_autotune.cu`. Probes top-3 candidates per cell; serves cached winner thereafter. Lock-free hot path. `FKC_AUTOTUNE=0` disables; `FKC_AUTOTUNE_VERBOSE=1` logs probes.
- **Force-route knobs**: `FKC_W4=1`, `FKC_WIDE3=1`, `FKC_NARROW=1`, `FKC_NTILES={1,2,4}`, `FKC_DSLAB=1`, `FKC_ASSIGN_DEEP_TILE=1` — bypass autotune for specific developer experiments.
- **Per-call env reads**: env knobs are read on every launch, so they take effect immediately (no process restart needed).

## Recommended first moves for the new session

1. Read `results.tsv` and this doc.
2. Rebuild clean source if a discarded experiment was compiled recently.
3. Run `scripts\windows\autotune_step.bat` to confirm baseline reproduces (~1043 TFLOPS when the GPU is in the same thermal/power state as the original run).
4. Run the verbose autotune (`FKC_AUTOTUNE_VERBOSE=1` ...) to see per-cell winners.
5. Do not start from dense `m16n8k32`, D-slab inter-slab pipelining, `N_TILES=3`, `N_TILES=5/6`, BK=16 S3/S4, `SMEM_PAD=4/16`, full-N specialization, `kProbeTopN=4`, or the simple two-pass split-K scratch prototype. Any next attempt needs a new structural idea or an integrated K-partitioning design without scratch materialization.
