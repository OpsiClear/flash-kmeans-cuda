# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

Two parallel implementations:

- `flash_kmeans_cuda/` — hand-rolled CUDA port (Phase A scope: hot Euclidean path only). Built as a torch C++/CUDA extension via `setup.py` / `pyproject.toml` at the repo root. Sources under `flash_kmeans_cuda/csrc/`. The mma.sync kernel in `csrc/assign/assign_sm80.cu` is **experimental** and gated behind `FKC_ASSIGN_USE_MMA=1`; the default fp16/bf16/fp32 path is the tensor-core-free `csrc/assign/assign_safe.cu`. See `flash_kmeans_cuda/ops.py` and `flash_kmeans_cuda/kmeans.py` for the Python surface.
- `third_party/flash-kmeans/` — upstream Triton implementation, vendored. Untouched; used as the correctness oracle in `tests/test_correctness.py`. Treat as a working copy of the upstream project.

Cosine, Dot, and `kmeans_largeN` are intentionally not in the CUDA port — callers can keep using `flash_kmeans` for those.

## Python environment

The user's global instruction is: **always use `uv run` for Python**. Don't invoke `python` / `pip` directly — run `uv run python ...`, `uv run pip install ...`, etc.

The CUDA port pins **torch 2.11.\*** built against **CUDA 13.0**, and **Python 3.12** (`requires-python = ">=3.12,<3.13"`). The cu130 wheel index is wired up in `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu130" }
```

`uv sync` / `uv run pip install -e .` will resolve torch from that index automatically. CPU-only torch will *not* satisfy the build because the extension links against the CUDA runtime.

Python ↔ C++ glue is **nanobind** (not pybind11). nanobind's `nb_combined.cc` is bundled into the extension build via `setup.py`; the `at::Tensor` caster lives in `flash_kmeans_cuda/csrc/nb_torch.h`. If you add new C++ ops, mirror that file's caster pattern — torch tensors don't auto-cast in nanobind.

Triton (used only by the upstream reference) is required for the fast path of `third_party/flash-kmeans`. On Windows the dependency is `triton-windows`, not `triton`. If Triton import fails the upstream package transparently falls back to a torch-native backend, so an "it ran" doesn't prove the Triton path was taken — check imports / warnings.

## Common commands

For the CUDA port (run from repo root). The Windows build is finicky and
requires the MSVC dev environment plus a few PATH tweaks — see the
`build_vc.bat` recipe below. On Linux just `uv sync` and `uv run pytest`.

**Windows build recipe** (write to a `.bat` file and run via `cmd //c`):

```batch
@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=%PATH:C:\Program Files\Git\usr\bin;=%"
set "PATH=%PATH:C:\Program Files\Git\mingw64\bin;=%"
set "PATH=%PATH:C:\Program Files\Git\usr\local\bin;=%"
set DISTUTILS_USE_SDK=1
cd /d C:\Users\HEQ\Projects\flash-kmeans-cuda
uv sync                                                  & rem first build
uv pip install -e . --no-build-isolation --reinstall-package flash-kmeans-cuda  & rem rebuild after kernel edits
uv run python -m pytest tests/
```

Why each step:
- `vcvars64.bat` puts MSVC's `cl`/`link` on PATH and sets `INCLUDE`/`LIB`.
- The `set "PATH=...=="` lines remove Git's `link.exe` (a GNU coreutils alias for `ln`) so distutils picks MSVC's linker.
- `DISTUTILS_USE_SDK=1` tells torch's CUDAExtension that the VC env is already configured.
- `--no-build-isolation` is critical for incremental rebuilds: it makes the build use the venv's already-installed cu130 torch, avoiding ABI mismatch with what the runtime loads.

**Common commands** (all paths after a successful build):

```bash
uv run python -m pytest tests/                  # full test suite
FKC_ASSIGN_FORCE_SAFE=1 uv run python -m pytest # debug: bypass mma path

uv run python benchmarks/bench_vs_triton.py \
    --batch-size 1 --num-points 32768 --dim 128 --num-clusters 256 --max-iters 5
```

For the upstream Triton reference (run from `third_party/flash-kmeans/`):

```bash
uv run python examples/testapi.py
uv run python examples/benchmark_backends.py \
    --batch-size 1 --num-points 32768 --dim 128 --num-clusters 256 --max-iters 3
uv run python -m flash_kmeans.kmeans_triton_impl
bash examples/run_benchmark_grid.sh
```

The CUDA port has a pytest suite under `tests/`; the upstream project does not.

### Triton runtime artifacts on this Windows host

`examples/benchmark_backends.py` redirects `TMP/TEMP/TMPDIR` and `TRITON_CACHE_DIR` into a local `.tmp/` and `.triton_cache/` next to the example, because Windows temp paths can break Triton's JIT. If you write new entry points that call Triton kernels, mirror this pattern (see the top of `examples/benchmark_backends.py`) or you may hit cryptic Triton compile errors. Both directories are gitignored.

## Architecture

### CUDA port (`flash_kmeans_cuda/`)

Three kernels behind a Python iteration driver that mirrors `kmeans_triton_impl.py:55-110`:

- `csrc/assign/assign_safe.cu` — Euclidean assignment for fp32 and fallback for fp16/bf16. One thread per (batch, point); streams K with fp32 accumulation. Tensor-core-free. Used by default for fp32; used as fallback for fp16/bf16 when the SMEM budget for the mma kernel doesn't fit (D=256 on Ada).
- `csrc/assign/assign_sm80.cu` — **default** m16n8k16 mma.sync kernel for fp16/bf16. Both A and B operand registers are populated via `ldmatrix.x4` (no `.trans`). The trick on B: viewed as a (rows=N, cols=K) matrix in our `(BLOCK_K, D)` row-major SMEM, ldmatrix.x4's post-load distribution `(N=t/4, K=2*(t%4)..+1)` is exactly what mma B expects, and a single `ldmatrix.x4` covers two N-atoms at once via the bit-3 (N-half) / bit-4 (K-half) lane partition. SMEM rows are padded by `SMEM_PAD = 8` fp16 (16 bytes) to break the 32-way bank conflict that hits power-of-2 D values; the 16-byte pad also preserves cp.async's 16-byte alignment. The min-over-K reduction is fused: each mma's output is reduced into per-thread `Best` registers immediately, so the cross-product tile is never materialized to SMEM (the structural diff vs Triton's `tl.dot`).

The launcher compiles four kernel variants and picks the largest that fits the device's per-block dynamic SMEM budget:
- `wide × 8 warps × 3 stages`  — preferred for K ≥ 512 (more warps/SM = better latency hiding under SMEM-bound 1-CTA/SM occupancy on Ada)
- `wide × 4 warps × 3 stages`  — preferred for K < 512 (more M-atoms per warp = higher arithmetic intensity)
- `wide × 4 warps × 2 stages`  — fallback for D=192 etc. where 3-stage SMEM is tight
- `deep × 4 warps × 2 stages`  — fallback when wide doesn't fit; opt-in via `FKC_ASSIGN_DEEP_TILE=1`
- `safe` kernel — final fallback when no mma tile fits (D=256 on Ada)

Set `FKC_ASSIGN_FORCE_SAFE=1` to bypass mma entirely for debugging.

**Perf snapshot (RTX 4090 / sm_89, fp16, vs Triton heuristic, ms/iter, two-run avg)**:

| Shape (B, N, K, D) | flash_kmeans_cuda | triton | ratio |
|---|---|---|---|
| 1, 8K, 128, 64 | 0.40 | 0.48 | **1.20× faster** |
| 1, 32K, 256, 128 | 0.40 | 0.51 | **1.28× faster** |
| 1, 75K, 1024, 128 (SVG2) | 0.68 | 0.55 | 0.81× (1.24× slower) |
| 1, 131K, 2048, 128 | 1.49 | 1.05 | 0.70× (1.42× slower) |

Geometric mean ratio: ~1.0× (tied with Triton on average). Started at 9× behind. Wins on small/medium K from the fused min-over-K reduction (no SMEM round-trip for the cross-product tile), `ldmatrix.x4` on both A and B, and the 8-warp wide tile providing latency hiding under SMEM-bound 1-CTA/SM occupancy. Three-stage cp.async pipeline + `#pragma unroll 8` on the d-loop + register-cached x_sq trim per-iter overhead.

Optimization path tried for K=2048: 4-stage narrow tile (regression), persistent-warp multi-N-tile (insufficient SMEM), register-cached x_sq (no measurable change), dropped explicit `cp.async.wait_all` (no measurable change), tile autotuning across `{narrow_4, wide_3_w8, wide_3_w4, wide_2_w4, deep_2_w4}` variants. Closing the last 1.4× requires profile-driven work (Nsight Compute) on stall reasons — likely SMEM-swizzling instead of padding to enable 2 CTAs/SM, or CUTLASS-style register-blocked epilogue.
- `csrc/update/update_sorted.cu` — sorted-chunk centroid accumulator. Caller (`flash_kmeans_cuda/ops.py`) does `torch.sort` on cluster_ids per batch, gathers x rows, then this kernel walks BLOCK_N=256 sorted tokens per CTA emitting one atomicAdd per run × BLOCK_D feature chunks. Output: fp32 sums + int32 counts.
- `csrc/update/update_finalize.cu` — `new[b,k] = where(count > 0, sums / count, old)` cast to compute dtype. Trivial 1D grid.

Build: `setup.py` invokes `torch.utils.cpp_extension.CUDAExtension`, emitting `-gencode` flags for sm_80/86/89/90/100/120. Hopper wgmma (Phase B) will compile a `assign_sm90.cu` with `-arch=sm_90a` only.

PTX wrappers (`csrc/common/ptx.cuh`): `cp.async`, `ldmatrix.x4` and `.trans`, `mma.sync.m16n8k16` for fp16/bf16. Arch guards in `csrc/common/arch.cuh`.

### Triton reference (`third_party/flash-kmeans/`)

Three layers, smallest to largest scope:

**1. Triton kernels (`flash_kmeans/assign_euclid_triton.py`, `flash_kmeans/centroid_update_triton.py`).** The hot path. `assign_euclid_triton` is autotuned over a grid of `(BLOCK_N, BLOCK_K, num_warps, num_stages)`; it also has a SMEM-fitting fallback that prunes configs that exceed the device's dynamic shared-memory budget for non-fp16/bf16 dtypes. There's a `use_heuristic=True` shortcut that skips autotune and picks a hand-tuned config — that's what the public `batch_kmeans_Euclid` uses by default. `centroid_update_triton` provides both an atomic-add update and a "sorted" variant (`triton_centroid_update_sorted_euclid`) that's preferred in the iteration loop.

**2. In-memory batched K-Means (`flash_kmeans/kmeans_triton_impl.py`).** Wraps the kernels into `batch_kmeans_Euclid` / `_Cosine` / `_Dot`. Each iteration is `_euclid_iter(x, x_sq, centroids)` → `(new_centroids, shift, cluster_ids)`. There's a `COMPILE_FLAG` (default `False`) for `torch.compile` wrapping; leave it off unless you're benchmarking — turning it on changes shape-specialization behavior. Convergence is `shift < tol`; `tol=0.0` means "always run `max_iters`". A torch-native version of the same loop lives in `flash_kmeans/torch_fallback.py` for when Triton is unavailable.

**3. Large-N streaming with multi-GPU (`flash_kmeans/kmeans_large.py`).** For data that doesn't fit in VRAM. Input lives on **pinned CPU memory**; `kmeans_largeN` partitions blocks across GPUs, runs a double-buffered H2D-overlap-compute pipeline per GPU using two `work_streams` per GPU, then performs a manual gather-reduce-broadcast on a `reduce_stream` (no NCCL — partial sums are ~4 MB, copies via NVLink/D2D are faster than spinning up a collective). Per-iteration phases are tagged in code as `Phase 1: Init`, `Phase 2: Block processing`, `Phase 3: Gather-Reduce-Broadcast`, `Phase 4: Sync`. `device=None` triggers the multi-GPU path; passing a single device forces single-GPU. The `is_last_block` short-circuit lets single-GPU runs finalize centroids inside the kernel without a separate reduce step — beware when refactoring that path.

**Public surface (`flash_kmeans/interface.py`).** `FlashKMeans` is the faiss/sklearn-style class. It dispatches between three paths inside `train()`:

- CPU input with `N > chunk_size_data_cpu` → `kmeans_largeN` (multi-GPU streaming). Batched (`B>1`) is **not supported** on this path; the code asserts.
- Otherwise, `use_triton=True` → `batch_kmeans_Euclid` (Triton).
- Otherwise → `batch_kmeans_Euclid_torch_native` (chunked PyTorch fallback, controlled by `chunk_size_data` / `chunk_size_centroids`).

`predict()` mirrors the same three-way dispatch using `euclid_assign_triton` / `kmeans_largeN_assign` / `euclid_assign_torch_native_chunked`. Calling `predict` with a different batch size than `train` raises.

The package's `__init__.py` swallows Triton import failures and rebinds the public names to torch fallbacks (or to a stub that raises `ImportError` for things like `kmeans_largeN` that have no fallback). So `from flash_kmeans import batch_kmeans_Euclid` always succeeds — what you actually got depends on whether Triton imported.

## Conventions worth knowing before editing

- Tensor shapes: in-memory APIs use `(B, N, D)`; large-N APIs use `(N, D)` and add the batch dim internally. `cluster_ids` is `int32` from Triton paths but `int64` from some torch paths — don't assume.
- `centroids` are kept in compute dtype (often fp16/bf16) but accumulators (`centroid_sums`) are always fp32; the multi-GPU reduce also stages in fp32 before casting back.
- Empty clusters: the multi-GPU finalize keeps the **old** centroid for any `count == 0` cluster (see the `torch.where(mask, ...)` in `kmeans_large.py`). Don't "simplify" this to a plain divide.
