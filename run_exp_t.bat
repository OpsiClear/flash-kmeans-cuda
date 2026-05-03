@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "PATH=%PATH:C:\Program Files\Git\usr\bin;=%"
set "PATH=%PATH:C:\Program Files\Git\mingw64\bin;=%"
set "PATH=%PATH:C:\Program Files\Git\usr\local\bin;=%"
set DISTUTILS_USE_SDK=1
cd /d C:\Users\HEQ\Projects\flash-kmeans-cuda
if not exist .tmp mkdir .tmp
set "BUILD_LOG=.tmp\build.log"
set "TEST_LOG=.tmp\test.log"
echo === build ===
uv pip install -e . --no-build-isolation --reinstall-package flash-kmeans-cuda > "%BUILD_LOG%" 2>&1
if errorlevel 1 (echo BUILD_FAILED & type "%BUILD_LOG%" & exit /b 1)
echo === smoke tests ===
uv run python -m pytest tests/test_shapes.py tests/test_dtypes.py tests/test_mma_optin.py -q > "%TEST_LOG%" 2>&1
if errorlevel 1 (echo TEST_FAILED & type "%TEST_LOG%" & exit /b 2)
echo === bench mega vs triton ===
uv run python benchmarks/bench_assign_vs_triton.py --shape mega --check-accuracy
