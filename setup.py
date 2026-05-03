"""Build script for flash_kmeans_cuda.

Compiles a single torch CUDAExtension named ``flash_kmeans_cuda._C`` containing
the Euclidean assign + sorted centroid update + finalize kernels.

Python <-> C++ glue uses **nanobind** (not pybind11). nanobind ships a single
``nb_combined.cc`` translation unit alongside its headers; we add it to the
extension's source list and pass nanobind's include dir. A custom
``at::Tensor`` type caster lives in ``csrc/nb_torch.h`` since nanobind doesn't
ship one for torch tensors.

Per-architecture isolation:
- ``assign/assign_sm80.cu`` (mma path) compiles for sm_80, sm_86, sm_89,
  sm_90, sm_100, and sm_120 when supported by the installed CUDA toolkit.
- ``assign/assign_safe.cu`` and the update kernels compile for every targeted
  arch.

We list ``-gencode arch=compute_XX,code=sm_XX`` for every supported arch so the
resulting wheel is portable. ``-arch=native`` is intentionally NOT used.
"""

from __future__ import annotations

import os
from pathlib import Path

import nanobind
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ROOT = Path(__file__).parent.resolve()
CSRC = ROOT / "flash_kmeans_cuda" / "csrc"

NB_PKG = Path(nanobind.__file__).resolve().parent
NB_INCLUDE = nanobind.include_dir()
NB_SRC_DIR = NB_PKG / "src"
NB_COMBINED = NB_SRC_DIR / "nb_combined.cpp"
# nanobind bundles tsl::robin_map under ext/robin_map/include — add it
# explicitly so #include <tsl/robin_map.h> resolves.
NB_ROBIN_INCLUDE = NB_PKG / "ext" / "robin_map" / "include"
if not NB_COMBINED.exists():
    raise FileNotFoundError(
        f"Expected nanobind combined source at {NB_COMBINED}; "
        "ensure nanobind>=2.1 is installed."
    )
if not NB_ROBIN_INCLUDE.exists():
    raise FileNotFoundError(
        f"Expected nanobind robin_map include at {NB_ROBIN_INCLUDE}; "
        "your nanobind install may be missing bundled deps."
    )


def _gencode_flags() -> list[str]:
    """Per-architecture flags for the broad-arch sources."""
    archs = ["80", "86", "89", "90", "100", "120"]
    # Only emit gencode flags for archs the installed nvcc actually understands.
    # Unknown ones are filtered at compile time by setting TORCH_CUDA_ARCH_LIST
    # via environment, but we also fall back gracefully.
    env_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if env_arch:
        return []  # let torch pick from the env var
    flags: list[str] = []
    for a in archs:
        flags += ["-gencode", f"arch=compute_{a},code=sm_{a}"]
    return flags


def _common_nvcc_flags() -> list[str]:
    # nanobind requires C++17 minimum. CUDAExtension already injects the
    # right host-compiler runtime flag (-Xcompiler /MD on Windows, -fPIC on
    # POSIX), so we don't repeat it here — passing /MD raw to nvcc would
    # be parsed as a positional input file.
    #
    # CCCL in CUDA 13 requires the conforming MSVC preprocessor; pass
    # /Zc:preprocessor through to cl. To avoid the std-ambiguity bug in
    # torch/csrc/dynamo/compiled_autograd.h, kernel TUs include only
    # <ATen/ATen.h> (not <torch/torch.h>) — see common/torch_cuda_includes.h.
    flags = [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-lineinfo",
    ]
    if os.name == "nt":
        flags += ["-Xcompiler=/Zc:preprocessor"]
    else:
        flags.append("-Xcompiler=-fPIC")
    return flags


def _common_cxx_flags() -> list[str]:
    if os.name == "nt":
        # /bigobj is required because nanobind's combined TU produces large COFFs.
        # /Zc:preprocessor: required by CCCL in CUDA 13 / nanobind macros.
        return ["/O2", "/std:c++17", "/EHsc", "/bigobj", "/Zc:preprocessor"]
    return ["-O3", "-std=c++17", "-fPIC", "-fvisibility=hidden"]


sources = [
    str(CSRC / "bindings.cpp"),
    str(CSRC / "assign" / "assign_safe.cu"),
    str(CSRC / "assign" / "assign_sm80.cu"),
    str(CSRC / "assign" / "assign_policy.cu"),
    str(CSRC / "assign" / "assign_autotune.cu"),
    str(CSRC / "update" / "update_sorted.cu"),
    str(CSRC / "update" / "update_finalize.cu"),
    str(NB_COMBINED),
]

include_dirs = [
    str(CSRC),
    str(CSRC / "common"),
    str(CSRC / "assign"),
    str(CSRC / "update"),
    NB_INCLUDE,
    str(NB_ROBIN_INCLUDE),
]


ext = CUDAExtension(
    name="flash_kmeans_cuda._C",
    sources=sources,
    include_dirs=include_dirs,
    extra_compile_args={
        "cxx": _common_cxx_flags(),
        "nvcc": _common_nvcc_flags() + _gencode_flags(),
    },
)


setup(
    name="flash-kmeans-cuda",
    version="0.1.0",
    packages=["flash_kmeans_cuda"],
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
