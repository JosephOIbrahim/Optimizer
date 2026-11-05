@echo off
REM ===========================================================================
REM  Optimizer V9 launcher  ->  python -m v9.cli %*
REM
REM  Picks a Python interpreter in this order:
REM    0. D:\OptimizerV9\venv  (PREFERRED: torch/TRT/CUTLASS live off-OneDrive;
REM                             override root via %OPTIMIZER_V9_DATA%)
REM    1. <repo>\venv\Scripts\python.exe   (only if it actually runs)
REM    2. py -3.12                  (TRT/torch wheels target 3.10-3.12)
REM    3. C:\Python314\python.exe   (works for config tuning; no TRT wheels)
REM    4. python on PATH
REM
REM  The in-repo committed venv historically pointed at a vanished miniconda;
REM  the probe below skips it automatically when broken. Run
REM  `v9_optimizer.bat doctor` for the honest health report.
REM
REM  Replaces v8_optimizer_opus47.bat's role; the v8 .bat is left intact.
REM ===========================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PYEXE="
set "PYLAUNCH="

REM --- 0. off-OneDrive venv (PREFERRED: torch/TRT live here, no OneDrive churn)
set "DATA_VENV=D:\OptimizerV9\venv\Scripts\python.exe"
if defined OPTIMIZER_V9_DATA set "DATA_VENV=%OPTIMIZER_V9_DATA%\venv\Scripts\python.exe"
if exist "%DATA_VENV%" (
    "%DATA_VENV%" -c "import sys" >nul 2>&1
    if !errorLevel! equ 0 set "PYEXE=%DATA_VENV%"
)

REM --- 1. committed in-repo venv, but only if it launches -------------------
if not defined PYEXE (
    set "VENV_PY=%~dp0venv\Scripts\python.exe"
    if exist "!VENV_PY!" (
        "!VENV_PY!" -c "import sys" >nul 2>&1
        if !errorLevel! equ 0 set "PYEXE=!VENV_PY!"
    )
)

REM --- 2. py -3.12 launcher (preferred: TRT/torch wheels target 3.12) -------
if not defined PYEXE (
    py -3.12 -c "import sys" >nul 2>&1
    if !errorLevel! equ 0 set "PYLAUNCH=1"
)

REM --- 3. known-good system interpreter (3.14: config tuning only) ----------
if not defined PYEXE if not defined PYLAUNCH (
    if exist "C:\Python314\python.exe" set "PYEXE=C:\Python314\python.exe"
)

REM --- 4. python on PATH ----------------------------------------------------
if not defined PYEXE if not defined PYLAUNCH (
    where python >nul 2>&1
    if !errorLevel! equ 0 set "PYEXE=python"
)

if not defined PYEXE if not defined PYLAUNCH (
    echo [ERROR] No usable Python found. Install Python 3.12+ or rebuild venv.
    echo         Run: py -3.12 -m venv venv ^&^& venv\Scripts\activate ^&^& pip install -r requirements.txt
    exit /b 1
)

if defined PYLAUNCH (
    py -3.12 -m v9.cli %*
) else (
    "%PYEXE%" -m v9.cli %*
)
exit /b %errorLevel%
