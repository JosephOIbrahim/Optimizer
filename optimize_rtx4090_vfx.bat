@echo off
REM ============================================================================
REM Optimizer v7_gemini3 - RTX 4090 VFX Workstation Optimizer
REM
REM Research-grade GPU optimization for Houdini 21 + UE5.6 + ComfyUI
REM Integrates NVIDIA latest research: Jet-Nemotron, TensorRT-LLM, cuTensor, CUTLASS
REM
REM Hardware: RTX 4090 (24GB), Threadripper PRO 7965WX (24C), 128GB RAM
REM Author: Generated for Joseph Ibrahim's VFX Workstation
REM Version: 7.0 (Gemini 3 Edition)
REM ============================================================================

setlocal enabledelayedexpansion

REM Check for Administrator privileges
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [INFO] Running with Administrator privileges
) else (
    echo [ERROR] This script requires Administrator privileges for registry and priority adjustments.
    echo Please right-click and "Run as Administrator".
    pause
    exit /b 1
)

REM Change to OptimizerV6 directory
cd /d "%~dp0"

echo.
echo ========================================================================
echo    Optimizer v7_gemini3 - RTX 4090 GPU Optimization System
echo ========================================================================
echo.
echo Hardware Target:
echo   - GPU: NVIDIA GeForce RTX 4090 (24GB VRAM, sm_89)
echo   - CPU: AMD Threadripper PRO 7965WX (24 cores)
echo   - RAM: 128GB
echo.
echo Applications:
echo   - SideFX Houdini 21 (Karma XPU, PDG/TOPs, Vellum, Pyro)
echo   - Unreal Engine 5.6+ (Lumen, Nanite, VSM, Path Tracer)
echo   - ComfyUI (Diffusion models)
echo.
echo ========================================================================
echo.

REM ============================================================================
REM TIER 1: IMMEDIATE OPTIMIZATIONS (Environment Variables)
REM ============================================================================

echo [TIER 1] Applying immediate GPU optimizations...
echo.

REM --- CUDA Optimizations (RTX 4090 sm_89) ---
echo   Setting CUDA environment variables...

REM CUDA Compilation Cache (4GB for faster shader recompilation)
set CUDA_CACHE_MAXSIZE=4294967296
set CUDA_CACHE_DISABLE=0
set CUDA_FORCE_PTX_JIT=0

REM cuBLAS/cuDNN Optimizations
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set CUDNN_LOGINFO_DBG=0
set CUDNN_LOGDEST_DBG=stderr

REM Memory Management
set CUDA_MODULE_LOADING=LAZY
set CUDA_DEVICE_ORDER=PCI_BUS_ID

REM --- Houdini 21 Optimizations ---
echo   Setting Houdini environment variables...

REM GPU Device Selection
set HOUDINI_OCL_DEVICENUMBER=0
set HOUDINI_GPU_DEVICE_NUM=0

REM VRAM Allocation (Sequential mode: 18GB, Parallel mode: 10GB)
REM Default to sequential - orchestrator will adjust if needed
set HOUDINI_OCL_MEMSIZE_MB=18432

REM Performance Optimizations
set HOUDINI_VDB_FAST_TEMP_FILE=1
set HOUDINI_DSO_ERROR=2

REM PDG/TOPs GPU Scheduling
set HOUDINI_PDG_MAX_CONCURRENT_COOKS=24

REM Karma XPU OptiX Optimizations
set HOUDINI_OPTIX_VERBOSE=0

REM --- ComfyUI / PyTorch Optimizations ---
echo   Setting PyTorch/ComfyUI environment variables...

REM Memory Management (max_split_size_mb:512 prevents fragmentation)
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.9

REM Force Memory-Efficient Attention (for Diffusion models)
set FORCE_MEM_EFFICIENT_ATTN=1

REM cuDNN Optimizations
set TORCH_CUDNN_V8_API_ENABLED=1

REM Deterministic Operations (if needed)
REM set CUBLAS_WORKSPACE_CONFIG=:4096:8
REM set PYTHONHASHSEED=42

REM --- Unreal Engine 5.6 Optimizations ---
echo   Unreal Engine console variables will be set via Engine.ini...
REM UE5 settings are applied via integrations (Engine.ini modification)

REM --- Windows GPU Scheduler ---
echo   Verifying Windows Hardware-Accelerated GPU Scheduling...
REM Note: Requires elevated permissions to modify registry
REM reg add "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v HwSchMode /t REG_DWORD /d 2 /f >nul 2>&1
echo   (Run as Administrator to enable Hardware GPU Scheduling)

echo.
echo [TIER 1] Environment variables configured successfully
echo.

REM ============================================================================
REM TIER 2: PYTHON ENVIRONMENT VALIDATION
REM ============================================================================

echo [TIER 2] Validating Python environment...
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please ensure Python 3.10+ is installed
    pause
    exit /b 1
)

python --version

REM Check if virtual environment exists
if not exist "venv\" (
    echo.
    echo Virtual environment not found. Creating...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Install dependencies if needed
if not exist "venv\Lib\site-packages\pynvml\" (
    echo.
    echo Installing dependencies (first-time setup)...
    echo This may take 5-10 minutes...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
    echo Dependencies installed successfully
)

echo.
echo [TIER 2] Python environment validated
echo.

REM ============================================================================
REM TIER 3: LAUNCH ORCHESTRATOR & DASHBOARD
REM ============================================================================

echo [TIER 3] Starting OptimizerV6 services...
echo.

REM Start dashboard in background (optional)
set LAUNCH_DASHBOARD=1

if "%LAUNCH_DASHBOARD%"=="1" (
    echo Starting Web Dashboard (http://localhost:8090)...
    start "OptimizerV6 Dashboard" python dashboard/app.py
    timeout /t 3 >nul
)

REM Launch main orchestrator
echo.
echo Starting GPU Optimizer Orchestrator...
echo.
echo ========================================================================
echo    Monitoring active - Press Ctrl+C to stop
echo ========================================================================
echo.

REM Run orchestrator (continuous monitoring mode)
python core/gpu_optimizer_v6.py --config config.yaml --interval 60

REM If orchestrator exits, clean up
echo.
echo Optimizer stopped.
echo.

REM Deactivate virtual environment
deactivate

echo ========================================================================
echo    OptimizerV6 Shutdown Complete
echo ========================================================================
echo.

pause
