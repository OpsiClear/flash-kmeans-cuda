@echo off
REM Run one auto-tune experiment: rebuild, run tests (quick), bench big shape.
REM Output goes to /tmp/exp.log; the speedup line is grepped by the loop.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "PATH=%PATH:C:\Program Files\Git\usr\bin;=%"
set "PATH=%PATH:C:\Program Files\Git\mingw64\bin;=%"
set "PATH=%PATH:C:\Program Files\Git\usr\local\bin;=%"
set DISTUTILS_USE_SDK=1
cd /d C:\Users\HEQ\Projects\flash-kmeans-cuda
echo === build ===
uv pip install -e . --no-build-isolation --reinstall-package flash-kmeans-cuda > _build.log 2>&1
if errorlevel 1 (echo BUILD_FAILED & type _build.log & exit /b 1)
echo === tests (smoke: shapes + dtypes) ===
uv run python -m pytest tests/test_shapes.py tests/test_dtypes.py tests/test_mma_optin.py -q > _test.log 2>&1
if errorlevel 1 (echo TEST_FAILED & type _test.log & exit /b 2)
echo === bench (big) accuracy + speedup ===
uv run python benchmarks/bench_vs_pytorch.py --shape big --check-accuracy
