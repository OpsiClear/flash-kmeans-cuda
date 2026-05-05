@echo off
setlocal
REM One auto-tune iteration: build + measure metric. Writes single-line
REM TOTAL_TFLOPS=<float> on stdout for the loop to grep.

call "%~dp0_setup_env.bat" || exit /b 1
pushd "%~dp0\..\.." || exit /b 1

if not defined TORCH_CUDA_ARCH_LIST set "TORCH_CUDA_ARCH_LIST=8.9"
if not defined MAX_JOBS set "MAX_JOBS=1"
if not exist .autotune mkdir .autotune

set "BUILD_LOG=.autotune\build.log"
set "METRIC_LOG=.autotune\metric.log"

uv run --no-sync python setup.py build_ext --inplace > "%BUILD_LOG%" 2>&1
if errorlevel 1 (
  echo BUILD_FAILED
  popd
  exit /b 1
)

uv run --no-sync python scripts/windows/autotune_metric.py > "%METRIC_LOG%" 2>"%METRIC_LOG%.err"
if errorlevel 1 (
  echo METRIC_FAILED
  popd
  exit /b 2
)

type "%METRIC_LOG%"
popd
