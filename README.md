# flash-kmeans-cuda

CUDA kernels for batched Euclidean K-Means, focused on the fp16 assignment hot
path on NVIDIA GPUs. This project is a libtorch/PyTorch CUDA implementation that
mirrors the public `flash-kmeans` Euclidean API while replacing the Triton
assignment/update path with hand-written CUDA kernels.

The current tuning target is RTX 4090 / Ada (`sm_89`), fp16, `D=128`, large
`N` and `K`.

## Status

This is an optimization-oriented CUDA port. The Euclidean fp16 path is the main
supported fast path. Cosine, dot-product, and large-N CPU streaming are still
provided by the original Triton project, not this CUDA port.

Recent RTX 4090 fp16 results, median-style local runs:

| Shape `(B, N, K, D)` | Assign: CUDA | Assign: Triton | Assign Speedup |
|---|---:|---:|---:|
| `(1, 32K, 256, 128)` | 92 TFLOPS | 38 TFLOPS | 2.40x |
| `(1, 131K, 2048, 128)` | 171 TFLOPS | 120 TFLOPS | 1.42x |
| `(1, 262K, 4096, 128)` | 176 TFLOPS | 126 TFLOPS | 1.39x |
| `(1, 524K, 8192, 128)` | 185 TFLOPS | 126 TFLOPS | 1.46x |

End-to-end K-Means includes assignment, sorting, centroid update, and finalize,
so speedups are lower and vary by shape:

| Shape `(B, N, K, D)` | CUDA ms/iter | Triton ms/iter | Speedup |
|---|---:|---:|---:|
| `(1, 32K, 256, 128)` | 0.129 | 0.508 | 3.94x |
| `(1, 131K, 2048, 128)` | 0.489 | 1.056 | 2.16x |
| `(1, 262K, 4096, 128)` | 1.667 | 2.786 | 1.67x |
| `(1, 524K, 8192, 128)` | 6.179 | 9.157 | 1.48x |

Quality checks compare final objective/inertia, centroid drift, and label
disagreement against the Triton reference. The latest sampled exact inertia
deltas were within about `+/-0.03%` on the large fp16 shapes.

## Design

The implementation has three layers.

### Python Driver

`flash_kmeans_cuda/kmeans.py` provides:

```python
from flash_kmeans_cuda import batch_kmeans_Euclid
```

The public shape convention is `(B, N, D)` for points and `(B, K, D)` for
centroids. The loop mirrors the original Triton implementation:

1. compute centroid norms
2. assign each point to its nearest centroid
3. sort labels
4. accumulate centroid sums/counts
5. finalize new centroids

The loop preallocates buffers across iterations, skips centroid-shift work when
`tol <= 0`, and skips `x_sq` setup for the D=128 fp16 raw-distance assignment
path where `x_sq` is row-constant and cannot affect `argmin`.

### CUDA Kernels

`flash_kmeans_cuda/csrc/assign/assign_sm80.cu` is the main tensor-core
assignment kernel for fp16/bf16 on Ampere+ GPUs.

Key points:

- `mma.sync.m16n8k16` tensor-core tiles.
- fp16 accumulator path for fp16 input on Ada.
- `ldmatrix.x4` loads for both operands.
- `cp.async` staging of point and centroid tiles into shared memory.
- fused min-over-K reduction in registers, so the cross-product matrix is never
  materialized.
- D=128 raw-distance specialization for `K>=256`.
- persistent N-tile variants; default keeps `N_TILES=2` for med/big/huge and
  routes mega `K>=8192` to `N_TILES=4`.

`assign_safe.cu` is the non-tensor-core fallback for fp32 or unsupported shapes.

`update_sorted.cu` accumulates centroid sums/counts from sorted cluster IDs. For
fp16 D=128 and `K>=256`, the indexed update path consumes the sorted
permutation directly and avoids materializing a full sorted copy of `x`.

`update_finalize.cu` computes `new_centroid = sum / count`, preserving old
centroids for empty clusters.

### Build Surfaces

There are two build surfaces:

- Python extension: `setup.py` builds `flash_kmeans_cuda._C` with nanobind.
- C++ shared library: `CMakeLists.txt` builds `flash_kmeans_cuda.dll` /
  `libflash_kmeans_cuda.so` with a libtorch-based public API.

The C++ API is declared in:

```cpp
#include <flash_kmeans_cuda/flash_kmeans_cuda.h>
```

It accepts and returns `at::Tensor` objects.

## Installation

### Python Development Install

Requirements used for the current Windows development environment:

- Python 3.12
- CUDA toolkit compatible with the installed PyTorch wheel
- PyTorch `2.11.*` CUDA wheel
- MSVC Build Tools on Windows
- `uv`

From a fresh checkout:

```powershell
git clone https://github.com/OpsiClear/flash-kmeans-cuda.git
cd flash-kmeans-cuda
uv sync --locked --python 3.12
```

Build the Python extension on Windows:

```powershell
cmd /c run_exp_t.bat
```

Run tests:

```powershell
uv run python -m pytest tests/ -q
```

## User Interface / API

This project exposes library interfaces rather than a graphical UI.

Python usage:

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

### C++ Shared Library

The C++ build is documented in [docs/cpp_shared_library.md](docs/cpp_shared_library.md).

Short Windows build:

```powershell
$env:TORCH_CUDA_ARCH_LIST = "8.9"
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

For Linux releases, prefer an explicit ABI/versioned artifact such as
`linux-x86_64-torch2.11-cu13-sm80_86_89_90`. The shared library links against
libtorch, so consumers must match the Torch/CUDA runtime family and C++ ABI.

## Benchmarks

Assign-only benchmark:

```powershell
uv run python benchmarks/bench_assign_vs_triton.py --shape mega --check-accuracy
```

End-to-end benchmark:

```powershell
uv run python benchmarks/bench_vs_triton.py --batch-size 1 --num-points 524288 --num-clusters 8192 --dim 128 --max-iters 1 --dtype fp16
```

Quality comparison over several iterations:

```powershell
uv run python .autotune\quality_compare.py --shapes med big huge mega --iters 10 --dtype fp16 --sample-points 4096
```

Available benchmark shapes in the local scripts:

| Name | Shape `(B, N, K, D)` |
|---|---|
| `med` | `(1, 32768, 256, 128)` |
| `big` | `(1, 131072, 2048, 128)` |
| `huge` | `(1, 262144, 4096, 128)` |
| `mega` | `(1, 524288, 8192, 128)` |

## Debug and Tuning Flags

Environment flags used by the launcher:

- `FKC_ASSIGN_FORCE_SAFE=1`: force safe non-MMA assignment.
- `FKC_NTILES=1|2|4`: override persistent N-tile routing.
- `FKC_ASSIGN_DEEP_TILE=1`: force deep tile fallback.
- `FKC_NARROW=1`, `FKC_WIDE3=1`, `FKC_W4=1`: experimental tile variants.

These flags are mainly for benchmarking and correctness A/B checks.

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
