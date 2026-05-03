cmake_minimum_required(VERSION 3.24)

project(flash_kmeans_cuda LANGUAGES CXX CUDA)

option(FKC_BUILD_SHARED_LIB "Build the flash-kmeans-cuda shared library" ON)

if(NOT FKC_BUILD_SHARED_LIB)
  return()
endif()

find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # Keep the CMake default portable but not excessive. PyTorch's CMake package
  # also honors TORCH_CUDA_ARCH_LIST, which is the preferred user-facing knob.
  set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

set(FKC_CSRC "${CMAKE_CURRENT_SOURCE_DIR}/flash_kmeans_cuda/csrc")

add_library(flash_kmeans_cuda SHARED
  "${FKC_CSRC}/api.cpp"
  "${FKC_CSRC}/assign/assign_safe.cu"
  "${FKC_CSRC}/assign/assign_sm80.cu"
  "${FKC_CSRC}/update/update_sorted.cu"
  "${FKC_CSRC}/update/update_finalize.cu"
)

add_library(flash_kmeans_cuda::flash_kmeans_cuda ALIAS flash_kmeans_cuda)

target_compile_definitions(flash_kmeans_cuda
  PRIVATE FKC_BUILD_SHARED
)

target_include_directories(flash_kmeans_cuda
  PUBLIC
    "$<BUILD_INTERFACE:${FKC_CSRC}>"
    "$<BUILD_INTERFACE:${FKC_CSRC}/common>"
    "$<BUILD_INTERFACE:${FKC_CSRC}/assign>"
    "$<BUILD_INTERFACE:${FKC_CSRC}/update>"
    "$<INSTALL_INTERFACE:include>"
)

if(TORCH_CXX_FLAGS)
  separate_arguments(FKC_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(flash_kmeans_cuda PRIVATE ${FKC_TORCH_CXX_FLAGS})
endif()

target_compile_options(flash_kmeans_cuda
  PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-O3>"
    "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>"
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>"
)

if(MSVC)
  target_compile_options(flash_kmeans_cuda
    PRIVATE
      "$<$<COMPILE_LANGUAGE:CXX>:/O2>"
      "$<$<COMPILE_LANGUAGE:CXX>:/EHsc>"
      "$<$<COMPILE_LANGUAGE:CXX>:/bigobj>"
      "$<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>"
      "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>"
  )
else()
  target_compile_options(flash_kmeans_cuda
    PRIVATE
      "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
      "$<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>"
      "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>"
  )
endif()

target_link_libraries(flash_kmeans_cuda
  PUBLIC
    "${TORCH_LIBRARIES}"
    CUDA::cudart
)

set_target_properties(flash_kmeans_cuda PROPERTIES
  OUTPUT_NAME flash_kmeans_cuda
  WINDOWS_EXPORT_ALL_SYMBOLS OFF
)

include(GNUInstallDirs)
install(TARGETS flash_kmeans_cuda
  EXPORT flash_kmeans_cudaTargets
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)

install(FILES "${FKC_CSRC}/flash_kmeans_cuda.h"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/flash_kmeans_cuda"
)

install(EXPORT flash_kmeans_cudaTargets
  NAMESPACE flash_kmeans_cuda::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/flash_kmeans_cuda"
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/flash_kmeans_cudaConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/flash_kmeans_cudaConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/flash_kmeans_cuda"
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/flash_kmeans_cudaConfig.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/flash_kmeans_cuda"
)
