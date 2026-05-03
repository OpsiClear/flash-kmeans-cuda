# Assign D-Slab Tiling Kernel — Design

**Date:** 2026-05-03
**Scope:** New kernel template `assign_sm80_dslab_kernel` parallel to the existing `assign_sm80_kernel`. Targets D ∈ {192, 224, 256, 320, 384}.
**Target arch:** Ada (sm_89, RTX 4090). Hopper / sm_90+ TMA-based variants are out of scope.

## Motivation

The existing `assign_sm80_kernel` loads `BN × D` x_smem in one shot and streams BK centroids per K-chunk. At D≥192 this puts x_smem alone at 51 KB; the `100 KB/CTA` Ada SMEM cap then forces BK ≤ 96 with STAGES=2 — and in practice none of the Stage 2 `widek96_*_d192` / `wide_3_*_d192` variants in the catalog actually fit (each is ~128 KB). The autotuner verified empirically that for D ∈ {192, 224, 256, 320, 384} mega-K, the only feasible variant is `narrowk32_w4_n2_d*` (BN=64, BK=32). Result: D=128 K=8192 hits 202 TFLOPS but D=192 K=8192 plateaus at 131 TFLOPS — 35% below the same-arch peak despite higher arithmetic intensity per byte loaded.

Root cause: the kernel's SMEM footprint scales linearly with D. To break the cap, x_smem must become D-independent. **D-slab tiling** restructures x_smem (and c_smem) to hold one D-slab at a time, with the K-chunk inner loop iterating over slabs and accumulating partial cross-products in registers across slabs.

## Locked decisions

- **Approach:** parallel D-slab kernel. The existing `assign_sm80_kernel` keeps owning D ∈ {64, 96, 128}; D-slab targets D ∈ {192, 224, 256, 320, 384}.
- **D_SLAB granularity:** maximum 128 features per slab. Each target D template-specializes its own partition (a list of slab widths summing to D). Smaller granularity is rejected: D_SLAB=64 doubles slab count and erodes the cp.async/mma overlap; D_SLAB=32 is hostile to future hardware (Hopper TMA, sm_90+ wgmma both prefer larger contiguous transfers).
- **Per-D partitions:**
  - D=192 → [128, 64]
  - D=224 → [128, 96]
  - D=256 → [128, 128]
  - D=320 → [128, 128, 64]
  - D=384 → [128, 128, 128]
- **Tile config:** every D-slab variant uses BN=128, BK=128, WARPS=8, STAGES=2 — the same config that hits 202 TFLOPS at D=128 today. Because x_smem is now bounded by max-slab=128, this tile fits at every target D.
- **Variant set:** `dslab_w8_n2_d{192,224,256,320,384}` and `dslab_w8_d{192,224,256,320,384}` (NT=2 and NT=1 each), per dtype. **10 variants × 2 dtypes = 20 new instantiations.** The dispatcher tries N_TILES=2 first (lower per-CTA launch overhead at large N); N_TILES=1 is the fallback for small N where the extra inter-tile cp.async drain costs more than the launch amortization saves.
- **Raw-distance:** every D-slab variant is `RAW=true`. The non-raw form is not instantiated; the Python loop's async-c_sq path doesn't use these D values.

## Non-goals (deferred)

- **Strategy C (split-K reduction)** — orthogonal; spec separately if D-slab leaves the (small-N, mega-K) corner short.
- **Strategy A (cp.async direct-to-register for centroids)** — much higher risk; revisit after D-slab.
- **D-slab for D ∈ {64, 96, 128}** — strictly worse there (more inner-loop overhead with no SMEM win).
- **sm_90 / Hopper TMA + wgmma kernel** — separate port.
- **Removing the legacy widek96_w8_n2_d192 etc. catalog entries** — kept as Hopper-future fallback in the policy rows.

## Architecture

Three layers, layered on top of the existing infrastructure from the policy-restructure (commits up to `3e61d5b`).

### Layer 1 — Kernel template (`assign_sm80_dslab_kernel.cuh`, new)

```cpp
template <typename T,
          int BN, int BK, int WARPS, int STAGES,
          int N_TILES,
          int D_FULL,                          // 192, 224, 256, 320, 384
          std::array<int, 4> SLAB_WIDTHS,      // partition; trailing 0s ignored
          int SMEM_PAD_SLAB = 0,               // bank-conflict pad on slab buffers; default 0 to fit SMEM
          bool RAW_DIST = true>
__global__ void __launch_bounds__(WARPS * 32, 1)
assign_sm80_dslab_kernel(
    const T* __restrict__ x,            // (B, N, D)
    const T* __restrict__ centroids,    // (B, K, D)
    const float* __restrict__ x_sq,     // (B, N) — unused when RAW_DIST
    const float* __restrict__ c_sq,     // (B, K)
    int32_t* __restrict__ cluster_ids,  // (B, N)
    int B, int N, int K, int D);
```

`SLAB_WIDTHS` is a `constexpr std::array` so the slab-iteration loop fully unrolls. `SMEM_PAD_SLAB` defaults to 0 so the BK=128 STAGES=2 config fits the Ada cap; raise to 4 for A/B testing if bank conflicts dominate.

#### SMEM layout (per CTA, fp16)

The legacy kernel uses `SMEM_PAD=8` per row of x_smem and c_smem to avoid bank conflicts when D is a power of 2. With SLAB_MAX=128 (power of 2) the same concern exists, but the math is:

```
With SMEM_PAD=8 (legacy default):
  x_slab_smem : 128 * (128+8) * 2  =  34 KB
  c_slab_smem : 2 * 128 * (128+8) * 2 = 68 KB
  c_sq_smem   : 2 * 128 * 4          =  1 KB
  Total                              = 103 KB  ← OVER 100 KB Ada cap

With SMEM_PAD=0 (new for D-slab):
  x_slab_smem : 128 * 128 * 2     = 32 KB
  c_slab_smem : 2 * 128 * 128 * 2 = 64 KB
  c_sq_smem   : 2 * 128 * 4       =  1 KB
  Total                           = 97 KB  ✓ under cap
```

**Decision: D-slab variants use `SMEM_PAD=0` for the slab buffers.** Bank-conflict risk is mitigated by two factors specific to the D-slab layout: (1) the c_slab buffer is double-buffered via cp.async, so successive c-slab loads write to alternate stages and don't contend with reads; (2) the per-slab mma sequence reads x_slab and c_slab in a strided pattern that already maps across all 32 banks for SLAB_MAX=128. If a future profiling pass shows residual conflicts, raise SMEM_PAD to 4 (still fits: 128*132*2 + 2*128*132*2 + 1KB = 100,224 bytes, just barely) or move to BK=96 (saves 16 KB at the cost of reduced K-chunk arithmetic intensity). SMEM_PAD is a `constexpr int` template parameter on the D-slab kernel so this can be A/B'd without a kernel rewrite.

`SLAB_MAX = 128` for every D-slab variant. SMEM footprint is identical across D values — the achievable tile config doesn't degrade as D grows.

#### Inner-loop pseudocode

```
for n_tile in 0..N_TILES:
    init best[m] = {+INF, -1}
    for k_chunk in 0..K step BK:
        init partial_cross[m][n] = 0.0   // float[M_ATOMS_PER_WARP*2][N_ATOMS_PER_WARP]
        for slab_idx in 0..NUM_SLABS:    // unrolled at compile time
            slab_off = constexpr_sum(SLAB_WIDTHS[0..slab_idx])
            slab_w   = SLAB_WIDTHS[slab_idx]
            cp.async load x_slab_smem  <- x[n_tile, slab_off : slab_off+slab_w]
            cp.async load c_slab_smem  <- centroids[k_chunk, slab_off : slab_off+slab_w]
            if slab_idx == 0:
                cp.async load c_sq_smem  <- c_sq[k_chunk]
            cp.async wait_all
            __syncthreads()
            for d_step in 0..slab_w step BLOCK_D=16:    // unrolled
                for m, n: mma(x_slab[m, d_step], c_slab[n, d_step], partial_cross[m][n])
        // All slabs done — finalize this k_chunk's BN×BK distance matrix.
        for m, n in registers:
            dist = RAW_DIST
                 ? -2.0f * partial_cross[m][n]
                 : x_sq[m] + c_sq[n] - 2.0f * partial_cross[m][n]
            if dist < best[m].dist: best[m] = {dist, k_chunk_base + n}
    // Reduce best across warps (existing pattern from legacy kernel).
    write cluster_ids[n_tile_rows] = best[:].idx
```

#### Differences from legacy `assign_sm80_kernel`

| Aspect              | Legacy                                                | D-slab                                                 |
|---------------------|-------------------------------------------------------|--------------------------------------------------------|
| x_smem load         | Once at CTA start, holds full (BN, D)                 | Per-K-chunk, per-slab; holds (BN, slab_w)              |
| x_smem size         | BN × D × elt (51 KB at D=192, 96 KB at D=384)         | BN × SLAB_MAX × elt (32 KB constant)                   |
| c_smem stage        | Holds full (BK, D)                                    | Holds (BK, slab_w)                                     |
| Per-K-chunk syncs   | 1 (cp.async wait + sync after c_tile load)            | NUM_SLABS (one per slab; 2 at D=192, 3 at D=384)       |
| partial_cross carry | Reset per K-chunk (D loop is innermost)               | Reset per K-chunk, accumulates across slabs            |
| D_FIXED             | Template param                                        | Encoded by D_FULL + SLAB_WIDTHS partition              |
| Epilogue            | Same dist + best update                                | Identical                                              |

### Layer 2 — Variant factory extension (`assign_dslab_variants.h`, new)

A sibling of `assign_variants.h`'s `make_variant<>`. Wraps the D-slab kernel:

```cpp
template <typename T,
          int BN, int BK, int W, int S, int NT,
          int D_FULL, std::array<int,4> SLABS, bool Raw>
struct DSlabVariantSpec {
  static size_t smem(int /*D*/, size_t elt) {
    constexpr int SLAB_MAX = 128;
    return /* BN * (SLAB_MAX+SMEM_PAD) * elt + STAGES * BK * (SLAB_MAX+SMEM_PAD) * elt + STAGES * BK * 4 */;
  }
  template <class T_> static bool try_launch(LaunchCtx& c) {
    if (c.D != D_FULL) return false;
    if (smem(c.D, c.elt_sz) > c.smem_limit) return false;
    /* launch dslab kernel */
    return true;
  }
};

template <int BN, int BK, int W, int S, int NT,
          int D_FULL, std::array<int,4> SLABS, bool Raw = true>
constexpr Variant make_dslab_variant(const char* name);
```

`make_dslab_variant<>` returns a `Variant` with the same struct layout as `make_variant<>` — drops cleanly into the existing `PolicyRow` and `AutotuneCache`.

### Layer 3 — Catalog and policy table updates (`assign_policy.cu`)

Add after the existing Stage 2 catalog block:

```cpp
constexpr std::array<int,4> kPart192 = {128, 64, 0, 0};
constexpr std::array<int,4> kPart224 = {128, 96, 0, 0};
constexpr std::array<int,4> kPart256 = {128, 128, 0, 0};
constexpr std::array<int,4> kPart320 = {128, 128, 64, 0};
constexpr std::array<int,4> kPart384 = {128, 128, 128, 0};

constexpr Variant V_DSLAB_W8_N2_D192 = make_dslab_variant<128, 128, 8, 2, 2, 192, kPart192>("dslab_w8_n2_d192");
constexpr Variant V_DSLAB_W8_D192    = make_dslab_variant<128, 128, 8, 2, 1, 192, kPart192>("dslab_w8_d192");
// … same pattern for D=224, 256, 320, 384.
```

Update each per-D row to lead with the D-slab variants:

```cpp
constexpr PolicyRow kD192 = {
  &V_DSLAB_W8_N2_D192, &V_DSLAB_W8_D192,
  &V_WIDEK96_W8_N2_D192, &V_WIDEK96_W8_D192,    // legacy Stage 2 (Ada-infeasible; kept for sm_90 future)
  &V_NARROWK32_W4_N2_D192, &V_NARROWK32_W4_N2,
  &V_NARROWK32_W4, &V_NARROW_4,
};
// kD224, kD256, kD320, kD384 follow the same shape.
```

`kD64`, `kD96`, `kD128`, `kGenericFallback`, `kRowsN1`, `kRowsN4` are unchanged.

### Layer 4 — Build wiring (`setup.py`)

The new `assign_sm80_dslab_kernel.cuh` is included by `assign_policy.cu` (which already includes the existing `assign_sm80_kernel.cuh` for the catalog's address-of constraint). No new `.cu` source files; no setup.py changes.

The `assign_sm80_dslab_kernel` template definition lives in the .cuh, instantiated by the catalog entries. Same anonymous-namespace device helpers (`mma_atom`, `async_load_tile`, `store_csq_tile`) can be shared with the legacy kernel — they're already in `assign_sm80_kernel.cuh`'s anonymous namespace and visible to any TU that includes either header.

## Testing

- **Correctness coverage:** existing `tests/test_assign_dispatch_equiv.py::test_all_locked_d_values_dispatch` (16 cases at K=256, 8 D values × 2 dtypes) re-runs unchanged. The autotuner will pick the new D-slab variants for D ∈ {192, 224, 256, 320, 384}; the test passes if the per-dtype thresholds (2% fp16 / 5% bf16) hold against the Python fp32 reference.
- **Cross-kernel agreement test:** add `tests/test_assign_dslab_vs_narrow.py` — for each D ∈ {192, 224, 256, 320, 384}, run with `FKC_DSLAB=1` (a new env knob that hoists the D-slab variant) and with `FKC_NARROW=1` (legacy narrowk32). Assert `cluster_ids` agree on ≥97% of points (allow 3% drift across kernels with different cross-product accumulation orders). One test function, parametrized 5×2 = 10 cases.
- **Bench acceptance** (`bench_d_sweep.py --n 32768 --k 8192`):
  - D=128: within ±2% of pre-change 202 TFLOPS (regression guard on the unchanged path).
  - D=192: ≥ 180 TFLOPS (vs current 131). **Primary acceptance.**
  - D=256: ≥ 180 TFLOPS (vs current 149). **Primary acceptance.**
  - D=320: ≥ current 167 TFLOPS.
  - D=384: ≥ current 161 TFLOPS.
- **Small-N regression check:** `bench_d_sweep.py --n 2048 --k 8192 --d 192 256` — D-slab must beat narrowk32 (today's winner) by ≥10% even when launch overhead is amplified.
- **Compile-time:** 20 new instantiations (10 variants × 2 dtypes). Same template-machinery cost per instantiation as existing variants; expected ≤1.2× current `assign_policy.cu` build wall time.

## Open risks

1. **Per-slab cp.async/sync overhead may swamp the SMEM win.** D=384 incurs 3× the `cp.async.wait_all + __syncthreads()` calls per K-chunk vs legacy. If the projected ≥180 TFLOPS at D=192 doesn't materialize, the autotuner falls back to legacy narrowk32 — no correctness risk, just no win. Mitigation if first build measures slow: add a per-slab cp.async pipeline (overlap slab `s+1` load with slab `s` mma). Adds complexity to the inner loop; defer unless first measurement requires it.

2. **`SLAB_WIDTHS` as `std::array` template param requires C++17 NTTP support for class types** — works in nvcc 12+ with `--expt-relaxed-constexpr` (already in `setup.py`'s nvcc flags). If nvcc rejects the `std::array<int,4>` non-type template parameter, fall back to encoding the partition as four separate int template parameters (`int S0, int S1, int S2, int S3`). Mechanically equivalent; uglier but bulletproof.

3. **D=224 partition `[128, 96]` is the only one needing a 96-wide slab path.** The kernel's slab-width handling now has three distinct mma loop counts (8/6/4 d-steps for slab widths 128/96/64). All compile-time-known per template instantiation; no runtime branching. If D=224 turns out to be a low-priority shape for the user's workloads, dropping it simplifies the kernel from 3 slab widths to 2 (just 128 and 64).

4. **partial_cross register pressure.** At BN=128 BK=128 8 warps: each warp holds M_ATOMS_PER_WARP=2, N_ATOMS_PER_WARP=16, so partial_cross is 64 floats per warp = 2 floats per thread. Negligible — same state as the legacy kernel's per-K-chunk cross accumulator, just held longer (across slabs instead of zeroed per d-step).

5. **`V_DSLAB_*_D192` etc. catalog entries become dead code if the kernel underperforms across the board.** Cleanup path: revert the `kD192..kD384` row promotions and remove the V_DSLAB entries; the legacy narrowk32 path continues to work. ~50 lines of revertable churn.

## Out-of-scope cleanups (deferred to post-merge)

- The Stage 2 `V_WIDEK96_W8_*_D{192,224}` / `V_WIDE_3_W8_*_D{192,224,256}` catalog entries are SMEM-infeasible on Ada and stay as no-ops. They remain in the catalog as Hopper-future fallback rows. Removing them is straightforward but not blocking; revisit when a Hopper port lands.
- The autotuner's per-call `getenv` cost (~3-10 µs flagged in the policy-restructure code-quality review) is unaffected by this spec.
