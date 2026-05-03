@echo off
setlocal
REM Run one auto-tune experiment: rebuild, run tests (quick), bench big shape.
REM Set VS_VCVARS to override the Visual Studio environment script path.
if not defined VS_VCVARS set "VS_VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VS_VCVARS%" (
  echo Missing Visual Studio vcvars64.bat. Set VS_VCVARS to its full path.
  exit /b 1
)
call "%VS_VCVARS%" >nul 2>&1
set "PATH=%PATH:C:\Program Files\Git\usr\bin;=%"
set "PATH=%PATH:C:\Program Files\Git\mingw64\bin;=%"
set "PATH=%PATH:C:\Program Files\Git\usr\local\bin;=%"
set DISTUTILS_USE_SDK=1
pushd "%~dp0\..\.." || exit /b 1
if not exist .tmp mkdir .tmp
set "BUILD_LOG=.tmp\build.log"
set "TEST_LOG=.tmp\test.log"
echo === build ===
uv pip install -e . --no-build-isolation --reinstall-package flash-kmeans-cuda > "%BUILD_LOG%" 2>&1
if errorlevel 1 (echo BUILD_FAILED & type "%BUILD_LOG%" & popd & exit /b 1)
echo === tests (smoke: shapes + dtypes) ===
uv run python -m pytest tests/test_shapes.py tests/test_dtypes.py tests/test_mma_optin.py -q > "%TEST_LOG%" 2>&1
if errorlevel 1 (echo TEST_FAILED & type "%TEST_LOG%" & popd & exit /b 2)
echo === bench (big) accuracy + speedup ===
uv run python benchmarks/bench_vs_pytorch.py --shape big --check-accuracy
popd
