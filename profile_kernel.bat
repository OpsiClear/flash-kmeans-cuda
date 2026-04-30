@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "PATH=%PATH:C:\Program Files\Git\usr\bin;=%"
set "PATH=%PATH:C:\Program Files\Git\mingw64\bin;=%"
set "PATH=%PATH:C:\Program Files\Git\usr\local\bin;=%"
cd /d C:\Users\HEQ\Projects\flash-kmeans-cuda
"C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.0\ncu.bat" --kernel-name regex:assign_sm80_kernel --launch-skip 5 --launch-count 1 --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis --section WarpStateStats --target-processes all uv run python benchmarks/bench_vs_pytorch.py --shape big --rounds 3
