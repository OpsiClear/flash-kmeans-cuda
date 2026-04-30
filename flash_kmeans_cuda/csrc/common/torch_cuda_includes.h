#pragma once

// Single include point for the torch CUDA helpers used by every kernel TU.
//
// On Windows, something in the include chain (most likely a CUDA runtime
// header that pulls minwindef.h) defines `small` as `char` and
// `min`/`max` as macros. These collide with parameter names in
// torch/CUDACachingAllocator.h (`bool small`) once the conforming MSVC
// preprocessor is in effect (CUDA 13's CCCL requires it). Undef the offenders
// here, immediately before pulling in ATen/c10 CUDA headers.

#ifdef small
  #undef small
#endif
#ifdef min
  #undef min
#endif
#ifdef max
  #undef max
#endif

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
