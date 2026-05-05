@echo off
setlocal
REM Build, run smoke tests, then bench the mega assign shape.
REM Set VS_VCVARS to override the Visual Studio environment script path.
call "%~dp0_setup_env.bat" || exit /b 1
pushd "%~dp0\..\.." || exit /b 1
if not exist .tmp mkdir .tmp
set "BUILD_LOG=.tmp\build.log"
set "TEST_LOG=.tmp\test.log"
echo === build ===
uv pip install -e . --no-build-isolation --reinstall-package flash-kmeans-cuda > "%BUILD_LOG%" 2>&1
if errorlevel 1 (echo BUILD_FAILED & type "%BUILD_LOG%" & popd & exit /b 1)
echo === smoke tests ===
uv run python -m pytest tests/test_shapes.py tests/test_dtypes.py tests/test_mma_optin.py -q > "%TEST_LOG%" 2>&1
if errorlevel 1 (echo TEST_FAILED & type "%TEST_LOG%" & popd & exit /b 2)
echo === bench mega vs triton ===
uv run python benchmarks/bench_assign_vs_triton.py --shape mega --check-accuracy
popd
