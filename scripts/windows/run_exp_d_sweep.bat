@echo off
setlocal
REM Local auto-tune experiment runner for D-sweep assign TFLOPS.
REM
REM This intentionally builds only for the local Ada target by default. Broad
REM release builds can override TORCH_CUDA_ARCH_LIST before invoking this file.
REM Compiling every CUDA 13.x arch is too slow for edit/measure loops and can
REM stall nvcc on high arch front-end passes such as compute_120.

if not defined VS_VCVARS set "VS_VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VS_VCVARS%" (
  echo Missing Visual Studio vcvars64.bat. Set VS_VCVARS to its full path.
  exit /b 1
)

call "%VS_VCVARS%" >nul 2>&1
set "PATH=%PATH:C:\Program Files\Git\usr\bin;=%"
set "PATH=%PATH:C:\Program Files\Git\mingw64\bin;=%"
set "PATH=%PATH:C:\Program Files\Git\usr\local\bin;=%"

if not defined TORCH_CUDA_ARCH_LIST set "TORCH_CUDA_ARCH_LIST=8.9"
if not defined MAX_JOBS set "MAX_JOBS=1"
set "DISTUTILS_USE_SDK=1"

pushd "%~dp0\..\.." || exit /b 1
if not exist .autotune mkdir .autotune
if not exist .tmp mkdir .tmp

set "BUILD_LOG=.autotune\build-d-sweep.log"
set "BENCH_LOG=.autotune\bench-d-sweep.log"

echo === build arch=%TORCH_CUDA_ARCH_LIST% max_jobs=%MAX_JOBS% ===
uv run --no-sync python setup.py build_ext --inplace > "%BUILD_LOG%" 2>&1
if errorlevel 1 (
  echo BUILD_FAILED
  type "%BUILD_LOG%"
  popd
  exit /b 1
)

echo === bench D sweep ===
uv run --no-sync python benchmarks/bench_d_sweep.py --n 32768 --k 8192 --d 128 192 224 256 320 384 --rounds 30 --warmup 5 --outer 3 > "%BENCH_LOG%" 2>&1
if errorlevel 1 (
  echo BENCH_FAILED
  type "%BENCH_LOG%"
  popd
  exit /b 2
)

type "%BENCH_LOG%"
popd
