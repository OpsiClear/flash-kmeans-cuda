@echo off
REM Shared MSVC + CUDA environment setup. Sourced via `call` from sibling
REM run_exp*.bat / profile_kernel.bat scripts. Sets ERRORLEVEL=1 if vcvars
REM is missing so callers can short-circuit.
REM
REM Caller may pre-set VS_VCVARS to override the default Visual Studio path.
REM
REM PATH stripping below removes Git's bundled `link.exe` / `cmd` / `find`
REM utilities that would otherwise shadow the MSVC toolchain after vcvars64
REM augments PATH. Substitution-with-trailing-`;` only matches the verbatim
REM default install location; if Git is installed elsewhere the strip is a
REM no-op (and the caller will likely hit a /LTCG link error).

if not defined VS_VCVARS set "VS_VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VS_VCVARS%" (
  echo Missing Visual Studio vcvars64.bat. Set VS_VCVARS to its full path.
  exit /b 1
)

call "%VS_VCVARS%" >nul 2>&1
set "PATH=%PATH:C:\Program Files\Git\usr\bin;=%"
set "PATH=%PATH:C:\Program Files\Git\mingw64\bin;=%"
set "PATH=%PATH:C:\Program Files\Git\usr\local\bin;=%"
set "DISTUTILS_USE_SDK=1"
exit /b 0
