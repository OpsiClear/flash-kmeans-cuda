# C++ Shared Library Build

This repo can build a libtorch-backed dynamic library for non-Python C++
projects. The Python extension path in `setup.py` is unchanged.

## Build

PowerShell from the repository root:

```powershell
$env:TORCH_CUDA_ARCH_LIST = "8.9"  # RTX 4090. Adjust for other GPUs.
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
