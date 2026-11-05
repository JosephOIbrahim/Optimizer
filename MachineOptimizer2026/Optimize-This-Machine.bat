@echo off
setlocal enabledelayedexpansion
title MachineOptimizer 2026 - RTX 4090 VFX

REM ==== self-elevate: registry + Defender exclusions need Administrator ====
net session >nul 2>&1
if %errorlevel% NEQ 0 (
    echo Requesting Administrator privileges...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -ArgumentList '%*' -Verb RunAs"
    exit /b
)

cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1

echo ================================================================
echo   MachineOptimizer 2026  -  RTX 4090 . Threadripper 7965WX . VFX
echo ================================================================
echo.

REM ==== Python =====================================================
REM requirements.txt states this tool runs on the Python standard library
REM alone -- nvidia-smi, PowerShell and winreg. 'rich' is the single optional
REM dependency and only makes the tables prettier. So a virtualenv is an
REM optimisation, not a prerequisite, and creating one needs the network.
REM Order: existing venv first (fastest, isolated), then system Python.
set "PYEXE="
if exist "venv\Scripts\python.exe" set "PYEXE=venv\Scripts\python.exe"

if not defined PYEXE (
    py -3.13 -c "" >nul 2>&1 && set "PYEXE=py -3.13"
)
if not defined PYEXE (
    py -3.12 -c "" >nul 2>&1 && set "PYEXE=py -3.12"
)
if not defined PYEXE (
    python -c "" >nul 2>&1 && set "PYEXE=python"
)
if not defined PYEXE (
    echo [ERROR] No Python 3.11+ found.
    echo         Install from https://www.python.org/downloads/  ^(tick "Add to PATH"^)
    pause & exit /b 1
)

REM Prove the interpreter can actually reach the package before doing anything.
%PYEXE% -c "import optimizer.orchestrator" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] %PYEXE% cannot import optimizer.orchestrator.
    echo         Run this .bat from its own folder, or check that optimizer\ exists here.
    echo         Current folder: %CD%
    pause & exit /b 1
)
echo Using Python: %PYEXE%

REM 'rich' is optional. Say so once, never block on it.
%PYEXE% -c "import rich" >nul 2>&1
if errorlevel 1 (
    echo   note: 'rich' not installed -- plain-text output instead of tables.
    echo         Optional. To add it:  %PYEXE% -m pip install rich
)

REM ==== pass-through: Optimize-This-Machine.bat --apply / --watch / --status ====
if not "%~1"=="" (
    echo.
    echo ---- running: %* ------------------------------------------------
    %PYEXE% -m optimizer.orchestrator %*
    set "RC=%errorlevel%"
    echo.
    echo Exit code: !RC!
    pause
    exit /b !RC!
)

REM ==== no arguments: preview, then offer to continue ==================
echo.
echo ---- PREVIEW (nothing is changed yet) --------------------------
%PYEXE% -m optimizer.orchestrator --dry-run
echo ----------------------------------------------------------------
echo.

choice /C YN /T 15 /D Y /M "Start the continuous watchdog now (applies optimizations + monitors)"
if errorlevel 2 (
    echo Exiting without changes. Re-run any time.
    pause & exit /b 0
)

echo.
echo Starting watchdog. A restore point is written before the first change.
echo Press Ctrl+C to stop.  Run Restore-This-Machine.bat to revert everything.
echo.
%PYEXE% -m optimizer.orchestrator --watch --yes

echo.
echo Watchdog stopped. The machine keeps whatever was applied.
echo Run Restore-This-Machine.bat to revert everything.
pause
