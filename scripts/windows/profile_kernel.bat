@echo off
setlocal
REM Profile the assign kernel with Nsight Compute.
REM Set VS_VCVARS or NCU_BAT to override local tool paths.
if not defined NCU_BAT set "NCU_BAT=C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.0\ncu.bat"
if not exist "%NCU_BAT%" (
  echo Missing Nsight Compute ncu.bat. Set NCU_BAT to its full path.
  exit /b 1
)
call "%~dp0_setup_env.bat" || exit /b 1
pushd "%~dp0\..\.." || exit /b 1
"%NCU_BAT%" --kernel-name regex:assign_sm80_kernel --launch-skip 5 --launch-count 1 --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis --section WarpStateStats --target-processes all uv run python benchmarks/bench_vs_pytorch.py --shape big --rounds 3
popd
