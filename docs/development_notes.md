# Development Notes

This file captures maintainer and agent notes for working on this repository.

## Repository layout

Two parallel implementations:

- `flash_kmeans_cuda/` — hand-rolled CUDA port for the Euclidean hot path. Built as a torch C++/CUDA extension via `setup.py` / `pyproject.toml`, and as a libtorch shared library via `CMakeLists.txt`. Sources live under `flash_kmeans_cuda/csrc/`. `api.cpp` owns validation and dispatch for both the Python extension and the C++ shared library; `bindings.cpp` is only the nanobind adapter. The default fp16/bf16 path uses the SM80 tensor-core assignment kernel when a supported policy row fits, and falls back to `assign_safe.cu` for fp32 or unsupported shapes.
- `third_party/flash-kmeans/` — upstream Triton implementation, vendored. Untouched; used as the correctness oracle in `tests/test_correctness.py`. Treat as a working copy of the upstream project.

Cosine, Dot, and `kmeans_largeN` are intentionally not in the CUDA port — callers can keep using `flash_kmeans` for those.

## Python environment

Use `uv run` for Python commands in this repository so the locked CUDA/PyTorch
environment is active: `uv run python ...`. Use `uv pip install ...` for
package installs into the active project environment.

The CUDA port pins **torch 2.11.\*** built against **CUDA 13.0**, and **Python 3.12** (`requires-python = ">=3.12,<3.13"`). The cu130 wheel index is wired up in `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu130" }
```

`uv sync` / `uv pip install -e .` will resolve torch from that index automatically. CPU-only torch will *not* satisfy the build because the extension links against the CUDA runtime.

Python ↔ C++ glue is **nanobind** (not pybind11). nanobind's `nb_combined.cpp` is bundled into the extension build via `setup.py`; the `at::Tensor` caster lives in `flash_kmeans_cuda/csrc/nb_torch.h`. If you add new C++ ops, mirror that file's caster pattern — torch tensors don't auto-cast in nanobind.

Triton is used only by the upstream reference, and it is required for that reference's fast path. Native Windows Triton wheels are unreliable in this environment; use Linux/WSL for clean Triton comparisons. If Triton import fails the upstream package can fall back to a torch-native backend, so an "it ran" doesn't prove the Triton path was taken — check imports / warnings.

## Common commands

Run commands from the repository root. The Windows helper
`scripts/windows/_setup_env.bat` runs `vcvars64.bat`, removes Git's `link.exe`
from `PATH`, and sets `DISTUTILS_USE_SDK=1` so Torch's CUDAExtension uses the
active MSVC environment.

Initial environment:

```powershell
uv sync --locked --python 3.12
```

Fast Windows editable build for RTX 4090 / `sm_89`:

```powershell
$env:TORCH_CUDA_ARCH_LIST = "8.9"
cmd /c "scripts\windows\_setup_env.bat 1>nul 2>nul && uv pip install -e . --no-build-isolation"
```

Install `pytest` for local smoke and dispatch tests:

```powershell
$env:TORCH_CUDA_ARCH_LIST = "8.9"
cmd /c "scripts\windows\_setup_env.bat 1>nul 2>nul && uv pip install pytest"
```

Rebuild after CUDA/C++ edits:

```powershell
$env:TORCH_CUDA_ARCH_LIST = "8.9"
cmd /c "scripts\windows\_setup_env.bat 1>nul 2>nul && uv pip install -e . --no-build-isolation --reinstall-package flash-kmeans-cuda"
```

Focused validation:

```powershell
cmd /c "scripts\windows\_setup_env.bat 1>nul 2>nul && uv run --no-sync python -m pytest tests/test_assign_dispatch_equiv.py -q"
cmd /c "scripts\windows\_setup_env.bat 1>nul 2>nul && uv run --no-sync python -m pytest tests/test_shapes.py tests/test_dtypes.py tests/test_mma_optin.py -q"
```

Install the full `dev` extra when you need the upstream `flash-kmeans` oracle:

```powershell
$env:TORCH_CUDA_ARCH_LIST = "8.9"
cmd /c 'scripts\windows\_setup_env.bat 1>nul 2>nul && uv pip install -e ".[dev]" --no-build-isolation'
```

The `dev` extra installs the upstream reference package. Triton wheels are
unreliable on native Windows; use Linux/WSL for clean Triton comparisons.

Full suite after the required test dependencies are installed:

```powershell
cmd /c "scripts\windows\_setup_env.bat 1>nul 2>nul && uv run --no-sync python -m pytest tests/ -q"
```

Benchmarks:

```powershell
uv run --no-sync python benchmarks/bench_d_sweep.py --n 32768 --k 8192 --d 1 2 3 4 8 16 128 192 224 256 320 384 --rounds 30 --warmup 5 --outer 3
uv run --no-sync python benchmarks/bench_vs_pytorch.py --shape huge --rounds 20 --check-accuracy
```

Use `scripts/windows/run_exp*.bat` for one-command build/test/bench loops.

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

Main source responsibilities:

- `csrc/api.cpp`: central validation and dispatch for both Python and C++.
- `csrc/bindings.cpp`: nanobind module; delegates assignment to `api.cpp`.
- `csrc/assign/assign_sm80.cu`: fp16/bf16 SM80+ tensor-core assignment
  launcher.
- `csrc/assign/assign_sm80_kernel.cuh`: core MMA kernel template.
- `csrc/assign/assign_policy.cu`: constexpr variant catalog and static policy
  rows.
- `csrc/assign/assign_autotune.cu`: in-memory first-call autotuner over the
  top feasible candidates per `(dtype, D, K bucket)` cell.
- `csrc/assign/assign_safe.cu`: fp32 path and exact/safe fallback, including
  small-D scalar/sorted helpers.
- `csrc/update/update_sorted.cu`: sorted centroid sum/count accumulator.
- `csrc/update/update_finalize.cu`: `sum/count` finalize with empty-cluster
  preservation.

Default assignment dispatch:

- fp16/bf16 `D=3..15` and `D % 16 == 0` enter the SM80 policy dispatcher.
- `D=1` and `D=2` use safe small-D paths by default because tensor-core padded
  variants lose too many ties at very low dimension.
- fp32 uses the safe kernel.
- If no SM80 candidate fits the device shared-memory limit, dispatch falls back
  to `assign_safe.cu`.

The current optimized D set is `1..16, 64, 96, 128, 192, 224, 256, 320, 384`.
Other multiples of 16 can still route through the generic SM80 fallback chain
or the safe kernel, but they are not the primary tuning target.

Iteration-loop optimizations in `flash_kmeans_cuda/kmeans.py`:

- skips shift computation when `tol <= 0` and `verbose=False`;
- reuses assignment, sum, count, and centroid buffers across iterations;
- skips `x_sq` materialization for the fp16 D=128 raw-distance path when the
  selected assignment kernel cannot read it;
- uses indexed centroid update for fp16 D=128 and `K>=256` to avoid
  materializing a full sorted copy of `x`.

PTX wrappers live in `csrc/common/ptx.cuh`; architecture guards live in
`csrc/common/arch.cuh`.

For active tuning notes and current benchmark tables, see
`docs/superpowers/HANDOFF-d-size-tflops.md`.

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
