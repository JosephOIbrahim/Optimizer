@echo off
setlocal enabledelayedexpansion
title MachineOptimizer 2026 - Watchdog

REM ==== self-elevate: affinity re-asserts + thermal -lgc need Administrator ====
net session >nul 2>&1
if %errorlevel% NEQ 0 (
    echo Requesting Administrator privileges...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

REM This launcher lives in the Optimizer folder but drives the package in its project subfolder.
set "OPTDIR=C:\Users\User\OneDrive\Desktop\Optimizer\MachineOptimizer2026"
cd /d "%OPTDIR%"
chcp 65001 >nul
set PYTHONUTF8=1

echo ================================================================
echo   MachineOptimizer 2026  -  Continuous Watchdog
echo ----------------------------------------------------------------
echo   Re-asserts the profile, fences CCD affinity per workload, and
echo   governs GPU thermals as Houdini / UE / ComfyUI come and go.
echo   A restore point is kept.  Press Ctrl+C to stop.
echo ================================================================
echo.

REM ==== pick Python (prefer venv, else 3.13/3.12/default) ====
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe -m optimizer.orchestrator --watch
) else (
    set "PYLAUNCH="
    py -3.13 -c "" >nul 2>&1 && set "PYLAUNCH=py -3.13"
    if not defined PYLAUNCH ( py -3.12 -c "" >nul 2>&1 && set "PYLAUNCH=py -3.12" )
    if not defined PYLAUNCH ( python -c "" >nul 2>&1 && set "PYLAUNCH=python" )
    if not defined PYLAUNCH (
        echo [ERROR] Python 3.12+ not found. Install from https://www.python.org
        pause & exit /b 1
    )
    !PYLAUNCH! -m optimizer.orchestrator --watch
)

echo.
echo Watchdog stopped. Machine keeps the applied optimizations.
echo Run Restore-This-Machine.bat to revert everything.
pause
