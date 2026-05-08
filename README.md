# flash-kmeans-cuda

Version `0.1.0`.

CUDA kernels for batched Euclidean K-Means, focused on the assignment hot path
on NVIDIA GPUs. The Python API mirrors the public Euclidean API from
`flash-kmeans`, while the Euclidean assign/update/finalize loop is implemented
with hand-written CUDA kernels and a libtorch/nanobind extension.

The current development target is RTX 4090 / Ada (`sm_89`), Python 3.12,
Torch 2.11, CUDA 13, fp16/bf16 Euclidean assignment over large `N` and `K`.

## Status

This is an optimization-oriented CUDA port. The main supported fast path is
Euclidean K-Means on CUDA tensors. Cosine, dot-product, and large-N CPU
streaming remain in the upstream Triton project, not this CUDA port.

Current assignment coverage:

- fp16/bf16 tensor-core path for `D=3..15` and multiples of 16.
- Specialized small-D handling for `D=1..16`, including exact/safe fallback
  paths where tensor-core tie behavior is too noisy.
- Tuned policy rows for `D=64, 96, 128, 192, 224, 256, 320, 384`.
- Generic safe fallback for fp32 and unsupported shapes.
- Per-process in-memory autotuning over candidate kernel variants, enabled by
  default.

Current local RTX 4090 fp16 D=128 assign-only numbers, measured from this tree
with `TORCH_CUDA_ARCH_LIST=8.9`:

| Shape `(B, N, K, D)` | CUDA ms | CUDA TFLOPS | PyTorch ms | Speedup vs PyTorch |
|---|---:|---:|---:|---:|
| `(1, 32K, 256, 128)` | 0.0245 | 87.54 | 0.2829 | 11.53x |
| `(1, 131K, 2048, 128)` | 0.4290 | 160.20 | 15.0723 | 35.14x |
| `(1, 262K, 4096, 128)` | 1.5635 | 175.81 | 59.1482 | 37.83x |
| `(1, 524K, 8192, 128)` | 6.3678 | 172.67 | n/a | n/a |

Current local `N=32768, K=8192` D-sweep sample:

| D | CUDA ms | CUDA TFLOPS |
|---:|---:|---:|
| 1 | 0.0592 | 9.07 |
| 2 | 0.1117 | 9.61 |
| 3 | 0.2118 | 7.60 |
| 4 | 0.1086 | 19.77 |
| 8 | 0.0934 | 46.01 |
| 16 | 0.0888 | 96.72 |
| 128 | 0.3935 | 174.66 |
| 192 | 0.6267 | 164.49 |
| 224 | 0.7311 | 164.50 |
| 256 | 0.9468 | 145.16 |
| 320 | 1.2132 | 141.61 |
| 384 | 1.3629 | 151.27 |

The D-sweep table is a single local sample. Re-run the commands below on a
quiet GPU before treating small differences as kernel wins or regressions.

End-to-end K-Means includes assignment, sorting, centroid update, and finalize,
so speedups are lower and vary by shape. Use the benchmark commands below for
numbers on your exact GPU, driver, CUDA toolkit, and D/K mix.

## Repository Layout

- `flash_kmeans_cuda/`: Python package and CUDA/C++ sources.
- `flash_kmeans_cuda/csrc/api.cpp`: shared C++ API validation and dispatch.
- `flash_kmeans_cuda/csrc/bindings.cpp`: nanobind Python adapter.
- `flash_kmeans_cuda/csrc/assign/`: assignment kernels, policy table, and
  autotuner.
- `flash_kmeans_cuda/csrc/update/`: sorted centroid update and finalize kernels.
- `benchmarks/`: assign, end-to-end, D-sweep, and quality comparison scripts.
- `tests/`: CUDA correctness, dispatch, autotune, shape, and dtype coverage.
- `scripts/windows/`: Windows environment setup, build, benchmark, and profiling
  helpers.
- `docs/`: C++ shared-library and maintainer notes.
- `third_party/flash-kmeans`: upstream Triton reference implementation.

## Python API

```python
import torch
from flash_kmeans_cuda import batch_kmeans_Euclid

x = torch.randn(1, 32768, 128, device="cuda", dtype=torch.float16)
labels, centroids, n_iters = batch_kmeans_Euclid(
    x,
    n_clusters=256,
    max_iters=10,
    tol=0.0,
)
```

Shape convention:

- points: `(B, N, D)`
- centroids: `(B, K, D)`
- labels: `(B, N)` int32

The loop mirrors the upstream Euclidean implementation:

1. compute centroid norms
2. assign each point to its nearest centroid
3. sort labels
4. accumulate centroid sums/counts
5. finalize new centroids

## Build Requirements

Pinned project requirements:

- Python `>=3.12,<3.13`
- Torch `2.11.*` from the cu130 wheel index
- CUDA toolkit compatible with the installed Torch wheel
- `nanobind>=2.1`
- `uv`

Windows build requirements:

- Visual Studio 2022 or Build Tools with MSVC C++ and Windows SDK
- NVIDIA CUDA toolkit on PATH
- PowerShell plus `cmd.exe`

Linux build requirements:

- GCC/Clang compatible with the installed CUDA toolkit
- NVIDIA CUDA toolkit
- A CUDA-capable PyTorch 2.11 cu130 install

## Python Development Build

From a fresh checkout:

```powershell
git clone --recursive https://github.com/OpsiClear/flash-kmeans-cuda.git
cd flash-kmeans-cuda
uv sync --locked --python 3.12
```

For an existing checkout:

```powershell
git submodule update --init --recursive
uv sync --locked --python 3.12
```

Fast local Windows rebuild for RTX 4090 / `sm_89`:

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

Portable/broad-arch builds can omit `TORCH_CUDA_ARCH_LIST`; `setup.py` then asks
Torch/NVCC to build the configured architecture set. That is much slower than a
single local-arch rebuild.

Linux editable build:

```bash
uv sync --locked --python 3.12
TORCH_CUDA_ARCH_LIST="8.9" uv pip install -e . --no-build-isolation
```

Run focused tests:

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
unreliable on native Windows; if Triton cannot import, the upstream package can
fall back to a torch-native backend. Use Linux/WSL for clean Triton comparisons.

Run the full test suite after the required test dependencies are installed:

```powershell
cmd /c "scripts\windows\_setup_env.bat 1>nul 2>nul && uv run --no-sync python -m pytest tests/ -q"
```

Convenience Windows scripts:

- `scripts\windows\run_exp.bat`: build, smoke tests, and PyTorch comparison.
- `scripts\windows\run_exp_t.bat`: build, smoke tests, and Triton assign bench.
- `scripts\windows\run_exp_d_sweep.bat`: build and D-sweep benchmark.

## C++ Shared Library

The C++ build is documented in
[docs/cpp_shared_library.md](docs/cpp_shared_library.md).

Short Windows build:

```powershell
$TorchPrefix = uv run python -c "import torch; print(torch.utils.cmake_prefix_path)"
cmake -S . -B build-shared -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH="$TorchPrefix" -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-shared --config Release --target flash_kmeans_cuda
cmake --install build-shared --config Release --prefix build-shared\install
```

External CMake project:

```cmake
find_package(Torch REQUIRED)
find_package(flash_kmeans_cuda CONFIG REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE flash_kmeans_cuda::flash_kmeans_cuda)
```

Example C++ call:

```cpp
#include <flash_kmeans_cuda/flash_kmeans_cuda.h>

auto ids = fkc::euclid_assign(x, centroids, x_sq, c_sq);
```

The C++ API accepts and returns `at::Tensor` objects. Consumers must match the
Torch/CUDA runtime family used to build the DLL/shared object.

## Benchmarks

Assign-only Triton comparison. This requires the upstream `flash-kmeans`
reference package and a working Triton install:

```powershell
uv run --no-sync python benchmarks/bench_assign_vs_triton.py --shape mega --check-accuracy
```

D sweep over the current tuning set:

```powershell
uv run --no-sync python benchmarks/bench_d_sweep.py --n 32768 --k 8192 --d 1 2 3 4 8 16 128 192 224 256 320 384 --rounds 30 --warmup 5 --outer 3
```

PyTorch reference comparison:

```powershell
uv run --no-sync python benchmarks/bench_vs_pytorch.py --shape huge --rounds 20 --check-accuracy
```

End-to-end comparison against upstream `flash_kmeans`. Verify imports if you
need the upstream Triton kernels rather than its torch fallback:

```powershell
uv run --no-sync python benchmarks/bench_vs_triton.py --batch-size 1 --num-points 524288 --num-clusters 8192 --dim 128 --max-iters 1 --dtype fp16
```

Quality comparison over several iterations. This imports upstream
`flash_kmeans`; without Triton it can exercise the upstream torch fallback
instead of the Triton kernels:

```powershell
uv run --no-sync python benchmarks/quality_compare.py --shapes med big huge mega --iters 10 --dtype fp16 --sample-points 4096
```

Available assign benchmark shapes:

| Name | Shape `(B, N, K, D)` |
|---|---|
| `med` | `(1, 32768, 256, 128)` |
| `big` | `(1, 131072, 2048, 128)` |
| `huge` | `(1, 262144, 4096, 128)` |
| `mega` | `(1, 524288, 8192, 128)` |

The PyTorch comparison scripts do not need Triton and work on the current
Windows development environment.

## Debug and Tuning Flags

Environment flags used by the assignment launcher:

- `FKC_ASSIGN_FORCE_SAFE=1`: force safe non-MMA assignment. Read per call.
- `FKC_AUTOTUNE=0`: disable the in-memory autotuner and use static policy order.
- `FKC_AUTOTUNE_VERBOSE=1`: print first-call probe timings.
- `FKC_NTILES=1|2|4`: override persistent N-tile routing.
- `FKC_NARROW=1`, `FKC_WIDE3=1`, `FKC_W4=1`: force developer tile variants.
- `FKC_ASSIGN_DEEP_TILE=1`: force the deep tile fallback candidate list.
- `FKC_DSLAB=1`: force the experimental D-slab path for supported large D.

These flags are for benchmarking and correctness A/B checks, not stable public
API.

## Release Automation

CI runs on pushes to `main` / `optimize/**`, on pull requests, and on manual
dispatch. It checks Python packaging, scans source distributions for generated
artifacts, and compiles both the Linux CUDA Python wheel and Linux C++ shared
library package.

GitHub Actions builds release artifacts when a `v*` tag is pushed:

```powershell
git tag v0.1.0
git push origin v0.1.0
```

The release workflow builds:

- Python source distribution
- Linux Python wheel for Python 3.12, Torch 2.11, CUDA 13.0
- Linux C++ shared-library package for Torch 2.11, CUDA 13.0, `sm80/86/89/90`

Manual rebuild/publish:

```powershell
gh workflow run release.yml -f tag=v0.1.0 -f publish=true
```

## Relationship to Flash-KMeans

This repository uses the upstream Triton project as the API and correctness
reference. The vendored reference lives under `third_party/flash-kmeans`.

Original project:

- GitHub: https://github.com/svg-project/flash-kmeans
- Paper: https://arxiv.org/abs/2603.09229
- Sparse VideoGen2 paper: https://arxiv.org/abs/2505.18875

If this CUDA port is useful in your work, cite the original Flash-KMeans work:

```bibtex
@article{yang2026flash,
  title={Flash-KMeans: Fast and Memory-Efficient Exact K-Means},
  author={Yang, Shuo and Xi, Haocheng and Zhao, Yilong and Li, Muyang and Fan, Xiaoze and Zhang, Jintao and Cai, Han and Lin, Yujun and Li, Xiuyu and Keutzer, Kurt and others},
  journal={arXiv preprint arXiv:2603.09229},
  year={2026}
}

@article{yang2025sparse,
  title={Sparse VideoGen2: Accelerate Video Generation with Sparse Attention via Semantic-Aware Permutation},
  author={Yang, Shuo and Xi, Haocheng and Zhao, Yilong and Li, Muyang and Zhang, Jintao and Cai, Han and Lin, Yujun and Li, Xiuyu and Xu, Chenfeng and Peng, Kelly and others},
  journal={arXiv preprint arXiv:2505.18875},
  year={2025}
}
```

## License

The upstream Flash-KMeans project is MIT licensed. Check the repository license
files before redistributing binaries that bundle or depend on PyTorch, CUDA, or
third-party components.
