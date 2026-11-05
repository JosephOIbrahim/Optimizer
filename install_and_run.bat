@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Optimizer V8 - Auto-Installer + Launcher

REM ============================================================
REM === Flag parsing ===========================================
REM ============================================================
REM Defaults: V8 mode on, TRT install attempted, dashboard launched.
REM   --v7              run legacy V7 orchestrator instead of V8
REM   --no-trt          skip TensorRT install attempt
REM   --no-dashboard    skip dashboard launch
REM   --build-profiler  build CUTLASS profiler (1-3 hr, ~10GB)
set "USE_V8=1"
set "INSTALL_TRT=1"
set "START_DASHBOARD=1"
set "BUILD_PROFILER=0"
for %%A in (%*) do (
    if /I "%%A"=="--v7" set "USE_V8=0"
    if /I "%%A"=="--no-trt" set "INSTALL_TRT=0"
    if /I "%%A"=="--no-dashboard" set "START_DASHBOARD=0"
    if /I "%%A"=="--build-profiler" set "BUILD_PROFILER=1"
)

color 0A
echo ========================================================
echo   Optimizer V8 - Auto-Installer + Launcher
echo   Target: RTX 4090 / sm_89 / Ada Lovelace
if "!USE_V8!"=="1" (
    echo   Mode:        V8 ^(gpu_optimizer_v8.py^)   [default]
) else (
    echo   Mode:        V7 ^(gpu_optimizer_v6.py^)   [--v7 override]
)
if "!INSTALL_TRT!"=="1" (
    echo   TRT:         install if missing             [default]
) else (
    echo   TRT:         skip                           [--no-trt]
)
if "!START_DASHBOARD!"=="1" (
    echo   Dashboard:   launch on http://127.0.0.1:8090 [default]
) else (
    echo   Dashboard:   skip                           [--no-dashboard]
)
echo ========================================================
echo.

REM Engine-cache dir up front (Forge-4's EngineCacheManager also creates).
if "!USE_V8!"=="1" (
    if not defined OPTIMIZER_V8_CACHE_DIR (
        set "OPTIMIZER_V8_CACHE_DIR=D:\OptimizerV8_EngineCache"
    )
    if not exist "!OPTIMIZER_V8_CACHE_DIR!" (
        echo [INFO] Creating V8 engine-cache directory: !OPTIMIZER_V8_CACHE_DIR!
        mkdir "!OPTIMIZER_V8_CACHE_DIR!" 2>nul
    )
)

REM Admin check (advisory, not fatal).
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARN] Not running as Administrator -- registry tweaks + priority
    echo        adjustments will be skipped. Right-click + "Run as Admin"
    echo        for full effect.
    echo.
    timeout /t 5 >nul
)

REM ============================================================
REM === Mile 1 of 9: Python ====================================
REM ============================================================
echo === Mile 1 of 9: Checking Python ===
python --version
if %errorLevel% neq 0 (
    echo [ERROR] Python not found on PATH. Install Python 3.10+ and rerun.
    pause
    exit /b
)

REM ============================================================
REM === Mile 2 of 9: Venv (create OR auto-rebuild if broken) ===
REM ============================================================
echo.
echo === Mile 2 of 9: Virtual Environment ===

set "VENV_OK=0"
if exist "venv" (
    REM Probe whether the venv's interpreter actually resolves. A stale
    REM shebang pointing at a removed base Python (miniconda etc.) is the
    REM common silent-fall-through bug -- detect it and rebuild instead
    REM of running the optimizer against unknown system packages.
    call venv\Scripts\activate.bat >nul 2>&1
    if !errorLevel! equ 0 (
        python --version >nul 2>&1
        if !errorLevel! equ 0 (
            set "VENV_OK=1"
            echo [INFO] Existing venv healthy.
        )
    )
)

if "!VENV_OK!"=="0" (
    if exist "venv" (
        echo [INFO] Existing venv is broken ^(stale shebang or missing base interpreter^).
        echo [INFO] Rebuilding from scratch...
        call venv\Scripts\deactivate.bat >nul 2>&1
        rmdir /s /q venv
    ) else (
        echo [INFO] No venv yet -- creating fresh.
    )
    python -m venv venv
    if !errorLevel! neq 0 (
        echo [ERROR] python -m venv failed. Check Python install.
        pause
        exit /b
    )
    call venv\Scripts\activate.bat
    if !errorLevel! neq 0 (
        echo [ERROR] Fresh venv activate failed -- environment is in an unexpected state.
        pause
        exit /b
    )
    python -m pip install --upgrade pip >nul 2>&1
)

REM ============================================================
REM === Mile 3 of 9: Python dependencies =======================
REM ============================================================
echo.
echo === Mile 3 of 9: Installing Python Dependencies ===
pip install -r requirements.txt
if %errorLevel% neq 0 (
    echo [ERROR] pip install -r requirements.txt failed.
    echo [HINT]  Common causes: Python version too new for some wheels,
    echo         or a transient PyPI hiccup. Retry, or pin Python 3.12.
    pause
    exit /b
)

REM ============================================================
REM === Mile 4 of 9: TensorRT (auto-install if missing) ========
REM ============================================================
echo.
echo === Mile 4 of 9: TensorRT ===
if "!INSTALL_TRT!"=="0" (
    echo [INFO] --no-trt: skipping TensorRT install.
    goto skip_trt
)
python -c "import tensorrt; print('  tensorrt', tensorrt.__version__)" 2>nul
if %errorLevel% equ 0 (
    echo [INFO] TensorRT already installed.
) else (
    echo [INFO] TensorRT not installed -- installing from NVIDIA index.
    echo        This is a ^~2GB download; first run takes several minutes.
    pip install tensorrt --extra-index-url https://pypi.nvidia.com
    if !errorLevel! neq 0 (
        echo [WARN] TensorRT install failed. V8 will run anyway --
        echo        TensorRTBackend will report enabled=False and TRT-gated
        echo        optimizations stay inactive. Rerun later or pip-install manually.
    ) else (
        python -c "import tensorrt; print('  tensorrt', tensorrt.__version__)"
    )
)
:skip_trt

REM ============================================================
REM === Mile 5 of 9: NVIDIA CUTLASS setup ======================
REM ============================================================
echo.
echo === Mile 5 of 9: NVIDIA CUTLASS ^(sm_89 / RTX 4090^) ===

REM Default path matches acceleration\cutlass_integration.py DEFAULT_PATH.
REM Override by exporting CUTLASS_PATH before running this script.
if not defined CUTLASS_PATH (
    set "CUTLASS_PATH=C:\Users\User\OneDrive\Desktop\Dads_Toolbox_For_Repairs\CUTLASS"
)
set "CUTLASS_INCLUDE_DIR=!CUTLASS_PATH!\include"

if exist "!CUTLASS_PATH!\include\cutlass\cutlass.h" (
    echo [INFO] CUTLASS already present at !CUTLASS_PATH!
) else (
    echo [INFO] CUTLASS not found - cloning NVIDIA/cutlass v3.5.1...
    where git >nul 2>&1
    if !errorLevel! neq 0 (
        echo [ERROR] git not found in PATH. Install Git for Windows
        echo         or set CUTLASS_PATH to an existing CUTLASS checkout.
        pause
        exit /b
    )
    for %%I in ("!CUTLASS_PATH!") do set "CUTLASS_PARENT=%%~dpI"
    if not exist "!CUTLASS_PARENT!" mkdir "!CUTLASS_PARENT!"
    git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git "!CUTLASS_PATH!"
    if !errorLevel! neq 0 (
        echo [ERROR] Failed to clone CUTLASS. Check network or set CUTLASS_PATH.
        pause
        exit /b
    )
    echo [INFO] CUTLASS v3.5.1 cloned to !CUTLASS_PATH!
)

echo [INFO] CUTLASS_PATH=!CUTLASS_PATH!
echo [INFO] CUTLASS_INCLUDE_DIR=!CUTLASS_INCLUDE_DIR!

if "!BUILD_PROFILER!"=="1" (
    echo.
    echo [INFO] --build-profiler: building CUTLASS profiler for sm_89 ^(1-3 hr, ~10GB^)
    timeout /t 5 >nul
    where cmake >nul 2>&1
    if !errorLevel! neq 0 (
        echo [WARN] cmake not found. Skipping profiler build ^(headers-only is fine for cutlass_integration.py except run_profiler^).
    ) else (
        if not exist "!CUTLASS_PATH!\build" mkdir "!CUTLASS_PATH!\build"
        pushd "!CUTLASS_PATH!\build"
        cmake .. -DCUTLASS_NVCC_ARCHS=89 -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_LIBRARY_KERNELS=all
        cmake --build . --config Release --target cutlass_profiler -j 24
        popd
    )
)

REM ============================================================
REM === Mile 6 of 9: Validate CUTLASS integration ==============
REM ============================================================
echo.
echo === Mile 6 of 9: Validating CUTLASS Integration ===
python -c "from acceleration.cutlass_integration import CUTLASSManager; m = CUTLASSManager(); s = m.get_status(); print(f'  installed: {s[\"installed\"]}'); print(f'  version:   {s[\"version\"]}'); print(f'  arch:      {s[\"target_arch\"]}'); print(f'  profiler:  {s[\"has_profiler\"]}')"
if %errorLevel% neq 0 (
    echo [WARN] CUTLASS validation hit an error - optimizer will run, but
    echo        CUTLASS-backed acceleration paths will be unavailable.
)

REM ============================================================
REM === Mile 7 of 9: ComfyUI presence check ====================
REM ============================================================
echo.
echo === Mile 7 of 9: ComfyUI presence ===
where curl >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARN] curl not found -- skipping ComfyUI HTTP probe.
    goto skip_comfy
)
curl -s -m 2 http://127.0.0.1:8188/system_stats >nul 2>&1
if %errorLevel% equ 0 (
    echo [INFO] ComfyUI is up on 127.0.0.1:8188.
    echo        V8's ComfyUI specialist + state collector will see it.
) else (
    echo [WARN] ComfyUI NOT detected on 127.0.0.1:8188.
    echo        V8 will run anyway -- specialists graceful-degrade -- but
    echo        R3 ^(compile-hot-shapes^) needs at least 3 generations of
    echo        the same model+resolution before firing. Start ComfyUI
    echo        ^(ideally with --use-sage-attention^) and run a workflow
    echo        a few times for full benefit.
)
:skip_comfy

REM ============================================================
REM === Mile 8 of 9: Dashboard =================================
REM ============================================================
echo.
echo === Mile 8 of 9: Dashboard ===
if "!START_DASHBOARD!"=="0" (
    echo [INFO] --no-dashboard: skipping dashboard launch.
    goto skip_dashboard
)
where curl >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARN] curl not found -- launching dashboard without pre-probe.
    goto launch_dashboard
)
curl -s -m 1 http://127.0.0.1:8090/ >nul 2>&1
if %errorLevel% equ 0 (
    echo [INFO] Dashboard already up at http://127.0.0.1:8090
    goto skip_dashboard
)
:launch_dashboard
echo [INFO] Starting dashboard in background ^(minimized window titled "Optimizer Dashboard"^)...
start "Optimizer Dashboard" /MIN cmd /c python dashboard\app.py

REM Wait up to 10s for dashboard to bind to 8090.
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
    echo        cmd window titled "Optimizer Dashboard" for errors. V8
    echo        will start anyway.
)
:skip_dashboard

REM ============================================================
REM === Mile 9 of 9: Run optimizer =============================
REM ============================================================
echo.
echo === Mile 9 of 9: Starting Optimizer ===
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
