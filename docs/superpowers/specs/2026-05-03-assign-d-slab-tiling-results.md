# Assign D-Slab Tiling — Task 9 Bench Results & Outcome

**Date:** 2026-05-03
**Reference spec:** `2026-05-03-assign-d-slab-tiling-design.md`
**Reference plan:** `2026-05-03-assign-d-slab-tiling.md`
**Final commit:** `cac1879` (Task 8 reverted)

## Headline outcome

**Stage 2 acceptance MISSED. Stage 1 acceptance (regression guard) PASSED.**

| D | Pre-D-slab | Post (autotuner picks narrowk32) | Force-dslab (FKC_DSLAB=1) | Plan target |
|---|---|---|---|---|
| 64 | 166 | 165 | n/a | (not targeted) |
| 96 | 198 | 198 | n/a | (not targeted) |
| **128** | 202 | **206** | n/a | within ±2% ✓ (regression guard) |
| **192** | 131 | 143 | **62** | ≥180 ✗ MISSED |
| 224 | 149 | 152 | 65 | ≥149 ✓ (soft) |
| **256** | 149 | 159 | **67** | ≥180 ✗ MISSED |
| 320 | 167 | 164 | 67 | ≥167 marginal |
| 384 | 161 | 168 | 67 | ≥161 ✓ (soft) |

(All numbers in TFLOPS at N=32768, K=8192, fp16, 30 rounds + 5 warmup.)

## What worked

1. **Kernel correctness.** D-slab kernel produces correct cluster_ids at every target D ∈ {192, 224, 256, 320, 384} for both fp16 and bf16. Cross-kernel agreement test (Task 7) showed <3% disagreement vs legacy narrowk32. Per-dtype thresholds in the equivalence suite (2% fp16 / 5% bf16) hold.
2. **SMEM math.** D-independent footprint achieved as designed. At BN=128, BK=128, STAGES=1, SLAB_MAX=128: ~65 KB total, well under the 100 KB Ada cap, regardless of D up to 384.
3. **Autotuner safety.** When dslab variants are in the policy rows, the autotuner correctly identifies them as too slow and falls back to legacy narrowk32. No regression on default-path behavior.
4. **Build infrastructure.** Variant factory, force-route knob, catalog entries, kernel scaffold — all the plumbing landed cleanly across 9 commits with full test coverage.

## What didn't work

**The D-slab kernel is 2-3× slower than legacy narrowk32 on Ada despite fitting more SMEM headroom.**

Root cause: the slab loop's per-slab `cp_async_wait_all() + __syncthreads()` kills cp.async / mma overlap. Every slab forces the kernel to drain all outstanding async loads before its mma can start, and to drain all again before the next slab can write to the same SMEM buffer (single-buffered c_slab_smem).

The plan's Open Risk #1 ("per-slab cp.async/sync overhead may swamp the SMEM win") materialized exactly as feared.

The Task 5 code-quality reviewer flagged this in a related form (PIPE_STAGES > 1 inert), and the chosen mitigation (drop to STAGES=1) addressed the SMEM-waste symptom but not the underlying pipeline-death cause. Without inter-slab pipelining (overlap slab N+1's load with slab N's mma using two SMEM stages), the kernel cannot match the legacy's overlap and so cannot beat its TFLOPS.

The legacy widek96/widek128 kernels, by contrast, load BK centroids per K-chunk in a 2-stage cp.async pipeline that overlaps load and compute end-to-end. They hit 200+ TFLOPS at D=128 BK=128 because of this overlap, not despite it.

## Action taken

**Reverted Task 8** (commit `cac1879`): the dslab variants are no longer in the per-D policy rows for D ∈ {192..384}. The autotuner sees only legacy variants for those cells.

**Kept everything else:**
- D-slab kernel (`assign_sm80_dslab_kernel.cuh`).
- Variant factory (`assign_dslab_variants.h`).
- Catalog entries (`V_DSLAB_*` in `assign_policy.cu`).
- Force-route knob (`FKC_DSLAB=1`).
- Smoke test + cross-kernel agreement test (`tests/test_assign_dslab.py`).

This preserves the kernel as a baseline for future inter-slab pipelining work without regressing default-path behavior.

## Path forward (next-cycle work, NOT in this spec)

To unlock the projected ≥180 TFLOPS at D=192/256, the dslab kernel needs **inter-slab cp.async pipelining**:

1. Allocate **two SMEM stages** for both x_slab_smem and c_slab_smem (4× the current allocation, but still ~130 KB max for STAGES=2 at SLAB_MAX=128 — actually fits if SMEM_PAD_SLAB=0).

   Wait — `2 × 32 KB (x) + 2 × 32 KB (c) + 2 × 0.5 KB (c_sq) = 129 KB` → over the 100 KB Ada cap.

   So either: (a) drop BN to 64 (halves x_slab to 16 KB → fits), or (b) drop BK to 64 (halves c_slab → fits).
   
   Option (a) — BN=64 BK=128 STAGES=2 dslab — looks viable: `2×16 + 2×32 + 2×0.5 = 97 KB`. Half the per-CTA throughput but full pipelining. May or may not beat legacy.
   
   Option (b) — BN=128 BK=64 STAGES=2 dslab — `2×32 + 2×16 + 2×0.25 = 96.5 KB`. Same total CTA work (BN×BK same), narrower K-chunks → 2× the K-chunk count but full pipelining.

2. Replace `cp_async_wait_all()` with `cp_async_wait_group<1>()` so the next slab's load can be in flight while the current slab's mma runs. Same pattern the legacy kernel uses across K-chunks; just applied across slabs within a K-chunk.

3. Bench-test both options against the current narrowk32 winner. Project: ≥150 TFLOPS at D=192 mega (50% lift from today's 142, halfway to the 180 target).

Estimated effort: 1-2 weeks. Risk: medium. The kernel restructure is bounded; the open question is whether the additional SMEM pressure from STAGES=2 forces tile-shape compromises that erode the pipeline win.

## Other follow-ups

- **Strategy C (split-K reduction)** — would compound with a pipelined D-slab to push small-N + large-K + large-D shapes further. Defer until pipelined dslab proves out.
- **Catalog cleanup** — the dslab variants in `assign_policy.cu` are dead code on Ada (autotuner won't pick them). Keep them so a future pipelined-dslab kernel can drop in by changing one template parameter (e.g., adding STAGES=2 + new wait_group logic). Cost: ~16 unused kernel instantiations × 32 KB each = ~500 KB binary size.
- **Stage 2 widek96_w8_n2_d192 etc. catalog entries** — also Ada-infeasible (128 KB > 100 KB cap), kept as Hopper-future fallback. Same cleanup decision.
