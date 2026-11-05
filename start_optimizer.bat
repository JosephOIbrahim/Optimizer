@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Optimizer V8 - Launcher

REM ============================================================
REM === start_optimizer.bat ====================================
REM ============================================================
REM Daily-use launcher. Assumes install_and_run.bat has already
REM provisioned venv + deps + CUTLASS at least once. Fails loudly
REM with a pointer to the installer if the venv isn't healthy.
REM
REM Flags:
REM   --v7              run legacy V7 orchestrator (default V8)
REM   --no-dashboard    skip dashboard launch (default on)
REM ============================================================

set "USE_V8=1"
set "START_DASHBOARD=1"
for %%A in (%*) do (
    if /I "%%A"=="--v7" set "USE_V8=0"
    if /I "%%A"=="--no-dashboard" set "START_DASHBOARD=0"
)

color 0A
echo ========================================================
echo   Optimizer V8 - Launcher
if "!USE_V8!"=="1" (
    echo   Mode:        V8 ^(gpu_optimizer_v8.py^)
) else (
    echo   Mode:        V7 ^(gpu_optimizer_v6.py^)
)
if "!START_DASHBOARD!"=="1" (
    echo   Dashboard:   http://127.0.0.1:8090
) else (
    echo   Dashboard:   skip
)
echo ========================================================
echo.

REM Engine-cache dir up front (matches install_and_run.bat default).
if "!USE_V8!"=="1" (
    if not defined OPTIMIZER_V8_CACHE_DIR (
        set "OPTIMIZER_V8_CACHE_DIR=D:\OptimizerV8_EngineCache"
    )
    if not exist "!OPTIMIZER_V8_CACHE_DIR!" (
        mkdir "!OPTIMIZER_V8_CACHE_DIR!" 2>nul
    )
)

REM Admin advisory.
net session >nul 2>&1
if !errorLevel! neq 0 (
    echo [WARN] Not running as Administrator -- registry tweaks + priority
    echo        adjustments will be skipped.
    echo.
)

REM ============================================================
REM === Mile 1 of 4: Activate venv =============================
REM ============================================================
echo === Mile 1 of 4: Activating venv ===
if not exist "venv" (
    echo [ERROR] No venv found at .\venv\
    echo [HINT]  Run install_and_run.bat first to provision the environment.
    pause
    exit /b
)
call venv\Scripts\activate.bat >nul 2>&1
if !errorLevel! neq 0 (
    echo [ERROR] venv activate failed.
    echo [HINT]  Run install_and_run.bat -- it will auto-rebuild a broken venv.
    pause
    exit /b
)
python --version >nul 2>&1
if !errorLevel! neq 0 (
    echo [ERROR] venv Python interpreter is broken ^(stale shebang or missing base^).
    echo [HINT]  Run install_and_run.bat -- it will auto-rebuild.
    pause
    exit /b
)
echo [INFO] venv healthy.

REM ============================================================
REM === Mile 2 of 4: ComfyUI presence ==========================
REM ============================================================
echo.
echo === Mile 2 of 4: ComfyUI presence ===
where curl >nul 2>&1
if !errorLevel! neq 0 (
    echo [INFO] curl not present -- skipping ComfyUI probe.
) else (
    curl -s -m 2 http://127.0.0.1:8188/system_stats >nul 2>&1
    if !errorLevel! equ 0 (
        echo [INFO] ComfyUI up on 127.0.0.1:8188.
    ) else (
        echo [WARN] ComfyUI NOT detected on 127.0.0.1:8188. V8 will run anyway.
        echo        R3 ^(compile-hot-shapes^) needs 3+ generations of the same
        echo        model+resolution to fire -- start ComfyUI when ready.
    )
)

REM ============================================================
REM === Mile 3 of 4: Dashboard =================================
REM ============================================================
echo.
echo === Mile 3 of 4: Dashboard ===
if "!START_DASHBOARD!"=="0" (
    echo [INFO] --no-dashboard: skipping dashboard launch.
    goto skip_dash
)

set "DASH_HAS_CURL=0"
where curl >nul 2>&1
if !errorLevel! equ 0 set "DASH_HAS_CURL=1"

if "!DASH_HAS_CURL!"=="1" (
    curl -s -m 1 http://127.0.0.1:8090/ >nul 2>&1
    if !errorLevel! equ 0 (
        echo [INFO] Dashboard already up: http://127.0.0.1:8090
        goto skip_dash
    )
)

echo [INFO] Starting dashboard in background ^(minimized window titled "Optimizer Dashboard"^)...
start "Optimizer Dashboard" /MIN cmd /c python dashboard\app.py

if "!DASH_HAS_CURL!"=="1" (
    set "DASH_OK=0"
    for /L %%i in (1,1,10) do (
        if "!DASH_OK!"=="0" (
            timeout /t 1 /nobreak >nul
            curl -s -m 1 http://127.0.0.1:8090/ >nul 2>&1
            if !errorLevel! equ 0 set "DASH_OK=1"
        )
    )
    if "!DASH_OK!"=="1" (
        echo [INFO] Dashboard up: http://127.0.0.1:8090
    ) else (
        echo [WARN] Dashboard didn't respond within 10s. Check the minimized
        echo        cmd window titled "Optimizer Dashboard" for errors.
    )
) else (
    echo [INFO] Dashboard launched ^(no curl, can't verify^). Try http://127.0.0.1:8090
)
:skip_dash

REM ============================================================
REM === Mile 4 of 4: Run optimizer =============================
REM ============================================================
echo.
echo === Mile 4 of 4: Starting Optimizer ===
echo.

if "!USE_V8!"=="1" (
    echo [INFO] Launching V8 orchestrator: core\gpu_optimizer_v8.py
    python core/gpu_optimizer_v8.py --config config.yaml --interval 60
) else (
    echo [INFO] Launching V7 orchestrator: core\gpu_optimizer_v6.py
    python core/gpu_optimizer_v6.py --config config.yaml --interval 60
)

echo.
echo [INFO] Optimizer finished.
echo [INFO] Dashboard window ^(if launched^) is still running -- close it manually.
endlocal
pause
