# Assign Kernel D-by-D Policy Restructure — Design

**Date:** 2026-05-03
**Scope:** `flash_kmeans_cuda/csrc/assign/assign_sm80.cu` dispatch logic (lines ~707–1154 today)
**Target arch:** Ada (sm_89, RTX 4090). Hopper / sm_90 dispatcher stays on the existing path until separately specced.

## Motivation

The current `launch_assign_sm80` dispatcher is a tangled if/else governed by one coarse threshold (`prefer_w8 = K >= 128`). Behavior gaps observed in the prior review:

1. Wide-tile + `D_FIXED` + raw-distance specialization exists only for `D=128`. Other D values fall through to the generic-D path or to `narrowk32` only, leaving measured perf on the table.
2. Env knobs (`FKC_NTILES`, `FKC_WIDE3`, `FKC_W4`, `FKC_NARROW`) are read once into `static const` at first launch — they cannot be flipped between calls in a single process, breaking per-shape A/B testing.
3. Variant selection is hand-tuned thresholds that have to be re-derived every time a new kernel is added.

This spec replaces the dispatcher with (a) a factory-based variant catalog, (b) a per-`(dtype, D, K_bucket)` static policy table, and (c) an in-memory autotuner cache that probes the top-3 static candidates on first call and caches the winner.

## Locked decisions

- **D set with full wide-tile specialization:** {64, 96, 128, 192, 224, 256, 320, 384}. Any other D falls through to a generic-D candidate list.
- **K-buckets (5):** `tiny: K<128 | small: <512 | med: <2048 | large: <8192 | mega: ≥8192`.
- **Autotune cache key:** `(dtype, D_idx, K_bucket)` — 2 × 9 × 5 = 90 cells max. In-memory only (no disk persistence).
- **Probe budget:** top-3 candidates per cold cell, 1 warmup + 3 timed runs per candidate, min of 3 wins.
- **Hot path:** lock-free read of cached winners after first probe per cell.

## Non-goals (deferred)

- Disk persistence of the autotune cache.
- `BN_bucket` axis on the cache key.
- Probing >3 candidates per cell.
- Hopper/sm_90 dispatcher changes.
- New kernel shapes beyond what already compiles today.

## Architecture

Three layers, all inside the existing `flash_kmeans_cuda/csrc/assign/` directory:

```
assign_variants.h     (new, header-only)   -- VariantSpec template + Variant struct + make_variant<…>
assign_policy.cu      (new)                -- constexpr Variant catalog + static_policy(dtype,D,K_bucket) table
assign_autotune.h     (new, header-only)   -- AutotuneKey, AutotuneCache, probe protocol
assign_sm80.cu        (modified)           -- launch_assign_sm80 reduces to: build LaunchCtx → autotune.get_or_probe → loop over candidates
```

### Layer 1 — variant factory (`assign_variants.h`)

One template per kernel shape. Each catalog entry is one `constexpr` line.

```cpp
struct LaunchCtx {
  const at::Tensor &x, &centroids, &x_sq, &c_sq;
  at::Tensor &cluster_ids;
  int B, N, K, D;
  size_t elt_sz;
  size_t smem_limit;
  bool async_csq;
  cudaStream_t stream;
};

template <int BN, int BK, int W, int S, int NT, int DFix=0, bool Raw=false>
struct VariantSpec {
  static size_t smem(int D, size_t elt) { return compute_smem_bytes(BN, BK, D, S, elt); }

  template <class T>
  static bool try_launch(LaunchCtx& c) {
    if (DFix && c.D != DFix) return false;
    if (smem(c.D, c.elt_sz) > c.smem_limit) return false;
    launch_typed_select_csq<T, BN, BK, W, S, NT, DFix, Raw>(
        c.x, c.centroids, c.x_sq, c.c_sq, c.cluster_ids,
        c.B, c.N, c.K, c.D, c.stream, c.async_csq);
    return true;
  }
};

struct Variant {
  const char* name;                                 // for FKC_AUTOTUNE_VERBOSE logs
  bool (*try_fp16)(LaunchCtx&);
  bool (*try_bf16)(LaunchCtx&);
  size_t (*smem)(int D, size_t elt);
};

template <int BN, int BK, int W, int S, int NT, int DFix=0, bool Raw=false>
constexpr Variant make_variant(const char* name) {
  using Spec = VariantSpec<BN, BK, W, S, NT, DFix, Raw>;
  return Variant{
    name,
    &Spec::template try_launch<__half>,
    &Spec::template try_launch<__nv_bfloat16>,
    &Spec::smem,
  };
}
```

### Layer 2 — variant catalog and policy table (`assign_policy.cu`)

**Variant catalog:** every kernel that compiles today gets one `constexpr Variant V_…` line. Stage 2 adds the missing combinations so each D ∈ {64, 96, 128, 192, 224, 256, 320, 384} has the wide D_FIXED + raw companion where SMEM permits.

Stage 2 instantiations to add (each gets a `RAW=false` and `RAW=true` companion):

| Variant family            | BN  | BK  | W | S | NT | New D values to add               |
|---------------------------|-----|-----|---|---|----|-----------------------------------|
| widek128_w8 (NT=1)        | 128 | 128 | 8 | 2 | 1  | 64, 96 (BK=128 too big for D≥192) |
| widek128_w8_n2            | 128 | 128 | 8 | 2 | 2  | (already 64, 96, 128)             |
| widek96_w8 (NT=1)         | 128 | 96  | 8 | 2 | 1  | 64, 96, 192, 224                  |
| widek96_w8_n2             | 128 | 96  | 8 | 2 | 2  | 64, 96, 192, 224                  |
| wide_3_w8 (NT=1)          | 128 | 64  | 8 | 3 | 1  | 192, 224, 256                     |
| wide_3_w8_n2              | 128 | 64  | 8 | 3 | 2  | 192, 224, 256                     |
| narrowk32_w4_n2           | 64  | 32  | 4 | 2 | 2  | (already 192, 224, 256, 320, 384) |

Cells whose `compute_smem_bytes` exceeds the 100 KB Ada cap are simply absent from the catalog (compile-time guarded by `if constexpr`).

**Policy table:**

```cpp
constexpr int N_DTYPES = 2;
constexpr int N_D      = 9;   // 64,96,128,192,224,256,320,384,OTHER
constexpr int N_KBKT   = 5;   // tiny, small, med, large, mega
constexpr int MAX_CAND = 6;

using PolicyRow = std::array<const Variant*, MAX_CAND>;  // null-terminated

constexpr std::array<std::array<std::array<PolicyRow, N_KBKT>, N_D>, N_DTYPES>
    kStaticPolicy = build_static_policy();
```

`build_static_policy()` is a `constexpr` function that fills cells from today's measured behavior — i.e., transcribes the existing if/else into table rows. Stage 3 reorders them at runtime; the static table is the cold-start default.

`OTHER` (any unsupported D) row uses today's generic fallback chain:
`[V_WIDEK128_W8, V_WIDEK96_W8, V_WIDE_3_W8, V_WIDE_3_W4, V_NARROW_4, V_DEEP_2_W4]`.

K-bucket helper:

```cpp
constexpr int k_bucket_of(int K) {
  if (K < 128)  return 0;
  if (K < 512)  return 1;
  if (K < 2048) return 2;
  if (K < 8192) return 3;
  return 4;
}
```

D-index helper maps the locked D set to indices 0..7 and everything else to 8 (OTHER).

### Layer 3 — autotuner (`assign_autotune.h`)

```cpp
struct AutotuneKey {
  int dtype_idx;  // 0=fp16, 1=bf16
  int d_idx;      // 0..8
  int k_bucket;   // 0..4
};

class AutotuneCache {
  struct Cell {
    std::array<const Variant*, MAX_CAND> ordered;  // reordered by probe; null-terminated
    std::atomic<bool> probed{false};
  };
  std::array<std::array<std::array<Cell, N_KBKT>, N_D>, N_DTYPES> cells_;
  std::mutex probe_mu_;

 public:
  std::span<const Variant* const> get_or_probe(AutotuneKey k, LaunchCtx& ctx,
                                                std::span<const Variant* const> static_candidates,
                                                bool autotune_enabled,
                                                bool verbose);
};
```

**Hot-path read (already probed):** acquire-load `cells_[k].probed`; if true, return `ordered`. No mutex.

**Cold-path probe (first miss):**

1. Acquire `probe_mu_`. Re-check `probed` (double-checked lock).
2. Filter `static_candidates` to the SMEM-feasible subset for `ctx.D` via `Variant::smem`.
3. Take top-3 of the filtered list (the static order is already a hand-tuned guess).
4. For each of the 3: warmup launch, then 3 timed launches with `cudaEvent_t` (record→record→sync→elapsedTime). Min of 3 wins.
5. Sort the 3 by measured time; write into `cells_[k].ordered` followed by the remaining (un-probed) feasible candidates as fallback. Null-terminate.
6. Release-store `probed = true`.

**Probe cost upper bound:** 3 candidates × 4 launches = 12 launches per cold cell. At ~1 ms/launch (mega K), ~12 ms per cell. All 90 cells worst case = ~1 s one-shot. Acceptable for kmeans training workloads.

**Correctness:** every probe launch runs the real kernel on the real `LaunchCtx`. `cluster_ids` is overwritten 12× and ends with the winner's output, identical to a non-probe single launch. No "probe mode" needed.

### Layer 4 — `launch_assign_sm80` (modified)

```cpp
void launch_assign_sm80(...) {
  // input checks unchanged …

  LaunchCtx ctx{x, centroids, x_sq, c_sq, cluster_ids, B, N, K, D,
                x.element_size(), smem_limit, K >= 256, stream};

  // Per-call config (formerly static-at-first-launch).
  EnvKnobs knobs = read_env_knobs();   // FKC_NTILES, FKC_WIDE3, FKC_W4, FKC_NARROW,
                                        // FKC_ASSIGN_DEEP_TILE, FKC_AUTOTUNE, FKC_AUTOTUNE_VERBOSE

  AutotuneKey key{ dtype_index(x), d_index(D), k_bucket_of(K) };

  std::span<const Variant* const> candidates;
  if (knobs.has_force_override()) {
    candidates = build_forced_candidates(knobs, ctx);   // bypasses autotune
  } else {
    candidates = autotune_cache().get_or_probe(
        key, ctx, kStaticPolicy[key.dtype_idx][key.d_idx][key.k_bucket],
        knobs.autotune, knobs.verbose);
  }

  bool launched = false;
  for (const Variant* v : candidates) {
    if (!v) break;
    bool ok = (x.scalar_type() == at::kHalf) ? v->try_fp16(ctx) : v->try_bf16(ctx);
    if (ok) { launched = true; break; }
  }
  if (!launched) launch_assign_safe(x, centroids, x_sq, c_sq, cluster_ids);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
```

`read_env_knobs()` is called every entry, not memoized — closes the static-at-first-launch gap. Per-call cost is a handful of `getenv` calls, negligible vs kernel launch.

`EnvKnobs::has_force_override()` returns true iff any of `FKC_NTILES`, `FKC_WIDE3`, `FKC_W4`, `FKC_NARROW`, or `FKC_ASSIGN_DEEP_TILE` is set. When true, `build_forced_candidates(knobs, ctx)` returns a fixed candidate list mirroring today's `force_*_env` branches in `assign_sm80.cu` (e.g., `FKC_W4=1` → `[V_WIDE_3_W4, V_WIDE_2_W4]`), ordered by the same precedence as today's if/else. The autotune cache is not consulted on these calls — the user has explicitly asked for a specific shape.

## Staging plan

**Stage 1 — Refactor (no behavior change).** Add `assign_variants.h` factory, `assign_policy.cu` with catalog of *only* today's existing variants, transcribe the current if/else into the static policy table. `launch_assign_sm80` becomes the new dispatch loop. No autotuner yet — the static table is consulted directly. Env knobs become per-call. **Acceptance:** every existing benchmark in `benchmarks/bench_assign_vs_triton.py` and `benchmarks/bench_d_sweep.py` produces identical (within ±2 % noise) timings vs the pre-refactor commit.

**Stage 2 — Fill gaps (perf change).** Add the new D-specialized instantiations from the table above. Update `assign_policy.cu` cells for D ∈ {64, 96, 192, 224, 256, 320, 384} to put the new variants ahead of the generic ones. **Acceptance:** `bench_d_sweep.py` shows non-regressing perf at D=128 and ≥ existing perf at all other D values; for at least one (D, K_bucket) cell outside D=128, perf improves measurably.

**Stage 3 — Autotuner.** Add `assign_autotune.h` and route the dispatch through it. Default `FKC_AUTOTUNE=1`. **Acceptance:** with `FKC_AUTOTUNE=1`, `bench_d_sweep.py` perf is ≥ Stage 2 at every (D, K) cell; verbose mode reports exactly one probe per `(dtype, D, K_bucket)` cell touched by the sweep, and zero probes on a second pass; `FKC_AUTOTUNE=0` reproduces Stage 2 numbers.

## Testing

- **Correctness:** existing pytest suite in `tests/` must pass unchanged at every stage. Add one new test that exercises every D in the locked set against the safe kernel reference.
- **Stage 1 numerical equivalence:** for a fixed seed, `cluster_ids` output must be bit-identical to the pre-refactor version on the same hardware. Verified by a one-off check, not a permanent test.
- **Stage 3 thread-safety:** two CUDA streams launching `assign` concurrently must not both enter the probe path for the same cell, and must not corrupt the cache. Verified by a small stress test (spawn 4 host threads, each calling assign on the same shape, assert exactly one probe occurred via verbose-log capture).
- **Compile-time:** Stage 2 adds ~64 new template instantiations; verify `setup.py build_ext` finishes within 1.5× current wall time. If not, drop to set (b) {64, 96, 128, 192, 256}.

## Open risks

1. **Compile-time blowup.** ~64 new instantiations (×2 dtypes ×2 raw-or-not = ~256 distinct kernels). Mitigation: incremental Stage 2 rollout (D=192,256 first; 224,320,384 second); if compile time spikes, keep narrowk32 fallbacks for the rare-D cases.
2. **`if constexpr` SMEM gating** must be tight — accidental instantiation of an over-budget kernel triggers nvcc errors at the launch site, not at the table. The factory's `smem()` is `constexpr` for known D, runtime for OTHER; ensure catalog excludes provably-too-big cells at compile time.
3. **Autotuner first-call latency** could surprise interactive notebook users (12 ms cold). Mitigation: `FKC_AUTOTUNE=0` opt-out documented in README.
4. **Reordered variants must remain SMEM-feasible.** The probe pre-filters by `Variant::smem`, but a reordered candidate list could still include an over-budget entry if the policy table seeds it incorrectly. Mitigation: `Variant::try_launch` re-checks SMEM and returns `false`; the dispatch loop just falls to the next candidate.
