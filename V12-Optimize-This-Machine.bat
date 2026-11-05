@echo off
setlocal enabledelayedexpansion
title Optimizer V12 - Gated Control Plane

REM ==== self-elevate: NVML clock pinning and any registry change need Administrator ====
REM Start-Process rejects an EMPTY -ArgumentList ("argument is null or empty"),
REM so a plain double-click (no arguments) must not pass one at all -- passing
REM '%*' unconditionally made the one-click launch die before it started.
net session >nul 2>&1
if %errorlevel% NEQ 0 (
    echo Requesting Administrator privileges...
    if "%~1"=="" (
        powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    ) else (
        powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -ArgumentList '%*' -Verb RunAs"
    )
    if errorlevel 1 (
        echo.
        echo [ERROR] Administrator rights were not granted, so V12 did not start.
        pause
    )
    exit /b
)

REM This launcher lives in the Optimizer folder and drives the package beside it.
set "V12DIR=C:\Users\User\OneDrive\Desktop\Optimizer\OptimizerV12"
cd /d "%V12DIR%"
chcp 65001 >nul
set PYTHONUTF8=1

echo ================================================================
echo   Optimizer V12  -  gated control plane
echo ----------------------------------------------------------------
echo   One loop:  OBSERVE - MODEL - DECIDE - ADMIT - ACT - REMEMBER
echo   No actuator moves state without a verdict. Dry run by default.
echo ================================================================
echo.

REM ==== Python: prefer an interpreter that can actually read the GPU =====
REM Two capabilities matter and they are independent:
REM   import v12     -- required
REM   import pynvml  -- live clocks, instantaneous power draw, power limits.
REM                     nvidia-smi covers everything else, so this is a
REM                     preference, never a requirement (see v12/truth.py).
REM Ordering by version number alone picked py -3.13 on this machine, which has
REM no pynvml, while the plain `python` (3.14, C:\Python314) does. Prefer
REM capability over recency, but never fail the launch over pynvml.
set "PYEXE="
set "PYEXE_NO_NVML="
for %%C in ("py -3.13" "py -3.12" "py -3.11" "python") do (
    %%~C -c "import v12" >nul 2>&1
    if not errorlevel 1 (
        if not defined PYEXE_NO_NVML set "PYEXE_NO_NVML=%%~C"
        %%~C -c "import pynvml" >nul 2>&1
        if not errorlevel 1 if not defined PYEXE set "PYEXE=%%~C"
    )
)
if not defined PYEXE set "PYEXE=!PYEXE_NO_NVML!"
if not defined PYEXE (
    echo [ERROR] No Python 3.11+ on PATH can import the v12 package.
    echo         Expected the package at: %V12DIR%\v12
    echo         Install Python from https://www.python.org/downloads/
    pause & exit /b 1
)
echo Using Python: %PYEXE%

REM Say which sensor path is live, because it changes what V12 can report:
REM without pynvml there are no live clocks and no instantaneous power draw --
REM the roofline is still correct, but the watchdog is blind to a throttling card.
%PYEXE% -c "import pynvml" >nul 2>&1
if errorlevel 1 (
    echo   note: pynvml not installed -- no live clocks or power draw.
    echo         V12 falls back to nvidia-smi: roofline stays correct, but it
    echo         cannot see the card throttling. Recommended, not required:
    echo             %PYEXE% -m pip install nvidia-ml-py
)

REM ==== pass-through ==============================================
if not "%~1"=="" (
    %PYEXE% -m v12 %*
    set "RC=!errorlevel!"
    echo.
    echo Exit code: !RC!
    pause
    exit /b !RC!
)

REM ==== no arguments: show the machine once, then loop on the menu ===
%PYEXE% -m v12 status
echo.
%PYEXE% -m v12 preflight

:menu
echo.
echo ----------------------------------------------------------------
echo   Everything below is read-only. Nothing on this machine changes.
echo.
echo   V12 applies a change only when THAT change was measured on this
echo   machine against a recorded baseline. Catalog entries that rest on
echo   a source (CITED) or are advice (GUIDED) are listed, never applied.
echo ----------------------------------------------------------------
echo.
echo   [1] Dry run   - every change, no workload  (see each decision)
echo   [2] Dry run   - diffusion  (FP8 / ComfyUI generation)
echo   [3] Dry run   - vellum     (Houdini solver)
echo   [4] Dry run   - render     (Karma XPU)
echo   [5] Record a baseline      (type in timings you measured)
echo   [6] Ledger    - what has been done, and how much was measured
echo   [7] Preflight - is anything else on the GPU right now?
echo   [8] Kernel toolchain readiness
echo   [9] Self-test - run V12's test suite
echo   [Q] Quit
echo.
choice /C 123456789Q /N /M "Choose: "

if errorlevel 10 goto :done
if errorlevel 9 goto :selftest
if errorlevel 8 goto :kernels
if errorlevel 7 goto :preflight
if errorlevel 6 goto :ledger
if errorlevel 5 goto :baseline
if errorlevel 4 set "OBJ=render"    & goto :dry
if errorlevel 3 set "OBJ=vellum"    & goto :dry
if errorlevel 2 set "OBJ=diffusion" & goto :dry
if errorlevel 1 goto :dryall

:dryall
echo.
echo ---- DRY RUN: all changes, no objective ------------------------
%PYEXE% -m v12 run
goto :pausemenu

:dry
echo.
echo ---- DRY RUN: %OBJ% -------------------------------------------
%PYEXE% -m v12 run --objective %OBJ%
echo.
echo To ACT on an admitted change, re-run with:  %~nx0 run --objective %OBJ% --apply
echo (--apply is a separate, explicit step on purpose.)
goto :pausemenu

:baseline
echo.
echo ---- RECORD A BASELINE -----------------------------------------
echo   A baseline is several timings of ONE workload, run the way it runs
echo   today, with nothing else heavy on the GPU (check option 7 first).
echo.
echo   [1] diffusion   seconds per image
echo   [2] vellum      seconds per frame
echo   [3] render      seconds per frame
echo   [B] back
echo.
choice /C 123B /N /M "Which workload: "
if errorlevel 4 goto :menu
if errorlevel 3 set "OBJ=render"    & goto :baseline_samples
if errorlevel 2 set "OBJ=vellum"    & goto :baseline_samples
if errorlevel 1 set "OBJ=diffusion" & goto :baseline_samples

:baseline_samples
set "SAMPLES="
echo.
set /p "SAMPLES=Timings for %OBJ%, comma-separated (e.g. 41.2,40.9,41.5) or Enter to cancel: "
if not defined SAMPLES goto :menu
%PYEXE% -m v12 baseline --objective %OBJ% --record "!SAMPLES!"
goto :pausemenu

:ledger
echo.
%PYEXE% -m v12 ledger
goto :pausemenu

:preflight
echo.
%PYEXE% -m v12 preflight
goto :pausemenu

:kernels
echo.
%PYEXE% -m v12 kernels
goto :pausemenu

:selftest
echo.
%PYEXE% -m unittest discover -s tests -q
goto :pausemenu

:pausemenu
echo.
pause
goto :menu

:done
endlocal
