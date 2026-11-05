@echo off
REM ============================================================================
REM  MachineOptimizer 2026 - tuned ComfyUI launcher (RTX 4090 / July 2026)
REM  Runs the validated D:\OptimizerV9 venv (Py3.12.10, torch 2.8.0+cu126)
REM  against C:\ComfyUI. Usage:
REM     Launch-ComfyUI-Optimized.bat            -> COMFYUI_ONLY  (--highvram)
REM     Launch-ComfyUI-Optimized.bat parallel   -> PARALLEL      (--normalvram)
REM  Also add this python.exe to the per-app 'No Sysmem Fallback' NVIDIA profile.
REM ============================================================================
setlocal

set "PY=D:\OptimizerV9\venv\Scripts\python.exe"
set "COMFY=C:\ComfyUI\main.py"

REM --- allocator + cache reuse (process-scoped, doesn't leak to other Python) ---
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "TORCHINDUCTOR_CACHE_DIR=D:\OptimizerV9\torchinductor_cache"
set "TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0"
set "CUDA_CACHE_PATH=D:\OptimizerV9\cuda_cache"

REM --- keep diffusion CPU work off the whole 48-thread machine (align to a CCD) ---
set "OMP_NUM_THREADS=8"
set "MKL_NUM_THREADS=8"

REM --- VRAM posture by mode ---
set "VRAMMODE=--highvram"
if /I "%~1"=="parallel" set "VRAMMODE=--normalvram"

echo Launching ComfyUI  [mode: %VRAMMODE%]
"%PY%" "%COMFY%" --disable-cuda-malloc --fast fp16_accumulation --use-sage-attention --reserve-vram 1.5 --cache-ram 8 %VRAMMODE%

pause
