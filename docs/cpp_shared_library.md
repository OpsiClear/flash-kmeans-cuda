# C++ Shared Library Build

This repo can build a libtorch-backed dynamic library for non-Python C++
projects. The shared library and Python extension both route through the same
`api.cpp` validation and dispatch layer.

## Requirements

- CMake 3.24+
- CUDA toolkit compatible with the Torch runtime
- Torch 2.11 CUDA install from this repo's `uv` environment
- MSVC/Visual Studio 2022 on Windows, or a CUDA-compatible GCC/Clang on Linux

Run this once before configuring CMake:

```powershell
uv sync --locked --python 3.12
```

## Build

PowerShell from the repository root:

```powershell
$TorchPrefix = uv run python -c "import torch; print(torch.utils.cmake_prefix_path)"
cmake -S . -B build-shared -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH="$TorchPrefix" -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-shared --config Release --target flash_kmeans_cuda
cmake --install build-shared --config Release --prefix build-shared\install
```

Build outputs:

- `build-shared/Release/flash_kmeans_cuda.dll`
- `build-shared/Release/flash_kmeans_cuda.lib`
- `build-shared/install/include/flash_kmeans_cuda/flash_kmeans_cuda.h`
- `build-shared/install/lib/cmake/flash_kmeans_cuda/*`

Linux:

```bash
TorchPrefix=$(uv run python -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake -S . -B build-shared -DCMAKE_PREFIX_PATH="$TorchPrefix" -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build-shared --target flash_kmeans_cuda -j
cmake --install build-shared --prefix build-shared/install
```

Use a semicolon-separated architecture list for portable builds, for example
`-DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"`.

## C++ Smoke Test

The repository includes a standalone C++ smoke test that uses libtorch tensors
directly and does not import the Python extension:

```powershell
$TorchPrefix = uv run python -c "import torch; print(torch.utils.cmake_prefix_path)"
cmake -S . -B build-shared -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH="$TorchPrefix" -DCMAKE_CUDA_ARCHITECTURES=89 -DFKC_BUILD_CPP_SMOKE=ON
cmake --build build-shared --config Release --target flash_kmeans_cuda_cpp_smoke
$env:PATH = "$PWD\build-shared\Release;$PWD\.venv\Lib\site-packages\torch\lib;$env:PATH"
.\build-shared\Release\flash_kmeans_cuda_cpp_smoke.exe
```

To test the installed package from a separate CMake project:

```powershell
$TorchPrefix = uv run python -c "import torch; print(torch.utils.cmake_prefix_path)"
$PackagePrefix = "$PWD\build-shared\install"
cmake -S tests\cpp\consumer -B build-shared\consumer-test -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH="$PackagePrefix;$TorchPrefix"
cmake --build build-shared\consumer-test --config Release --target consumer_smoke
$env:PATH = "$PackagePrefix\bin;$PWD\.venv\Lib\site-packages\torch\lib;$env:PATH"
.\build-shared\consumer-test\Release\consumer_smoke.exe
```

## Consume From Another CMake Project

```cmake
find_package(Torch REQUIRED)
find_package(flash_kmeans_cuda CONFIG REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE flash_kmeans_cuda::flash_kmeans_cuda)
```

Configure the consumer with both Torch and this package on `CMAKE_PREFIX_PATH`.
At runtime on Windows, put `flash_kmeans_cuda.dll` and the torch CUDA DLLs on
`PATH`, or copy them next to the executable.

```cpp
#include <flash_kmeans_cuda/flash_kmeans_cuda.h>

auto ids = fkc::euclid_assign(x, centroids, x_sq, c_sq);
```

The API accepts and returns `at::Tensor` objects, so consumers must link
against the same libtorch/CUDA runtime family used to build the DLL.
