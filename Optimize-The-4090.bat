@echo off
REM ===========================================================================
REM  OPTIMIZE THE 4090  -  next generation
REM
REM  One question, not a menu of knobs: what do you want faster?
REM  The tool measures the machine it is on, works out what is actually
REM  limiting the card for THAT request, and only then proposes a lever.
REM
REM  Everything is read-only until a lever is measured. Nothing stays applied.
REM  Power is raised only if the measurement says the card is pinned at its
REM  ceiling, and it is put back exactly afterwards.
REM
REM  Arguments are passed straight through to the tool, which owns the
REM  defaults and the parsing. That is deliberate: cmd.exe splits an argument
REM  on a COMMA before a batch file ever sees it, so "--levers a,b" arrives
REM  as %1=--levers %2=a %3=b. Forwarding %* verbatim keeps the text
REM  intact, and one parser is better than two that disagree.
REM ===========================================================================
setlocal enabledelayedexpansion
title OPTIMIZE THE 4090 - measure, don't believe
color 0F
chcp 65001 >nul

set "REPO=C:\Users\User\OneDrive\Desktop\Optimizer"
set "V12PY=C:\Python314\python.exe"
set "TORCHPY=E:\OptimizerV12\venv\Scripts\python.exe"
set "PYTHONUNBUFFERED=1"
cd /d "%REPO%"

REM --- pick an interpreter that can actually do the job ---------------------
REM The compute path needs torch+CUDA; the render path needs PIL/numpy for the
REM ssim guardrail. The 3.14 interpreter has all three (verified 2026-11-04:
REM torch 2.9.1+cu130, CUDA available, numpy, PIL, pxr). So it is the default
REM and the venv is the fallback.
set "PY=%V12PY%"
"%V12PY%" -c "import torch,numpy,PIL" >nul 2>&1
if errorlevel 1 if exist "%TORCHPY%" set "PY=%TORCHPY%"

if not exist "%V12PY%" (
    echo [ERROR] %V12PY% is missing. This tool needs a Python 3.11+ with torch.
    pause
    exit /b 1
)

REM --- was a request named on the command line? -----------------------------
REM Checked over the whole tail, not just %1, so the flags may appear in any
REM order. Without a request, the menu asks which one.
echo %* | findstr /i /c:"--request" >nul
if not errorlevel 1 goto :run

echo %* | findstr /i /c:"/?" >nul
if not errorlevel 1 goto :usage

if "%~1"=="" goto :menu
goto :run

:run
cls
"%PY%" "%REPO%\harness\optimize.py" %*
set "RC=!ERRORLEVEL!"
REM The tool commits the measurement record itself, at the end of the run,
REM in ONE commit that says what the campaign found. It used to be committed
REM here instead, in a second commit that only said "sync" -- which told a
REM reader of the history nothing about what had been measured.
exit /b !RC!

:menu
cls
echo.
echo   ================================================================
echo.
echo      O P T I M I Z E   T H E   4 0 9 0
echo.
echo      What do you want faster on this box?
echo.
echo      The card is not the constraint in every workload, so the tool
echo      measures first and only proposes levers that can pay.
echo.
echo   ================================================================
echo.
echo     [1] RENDER A SHOT      N frames of a Houdini/Karma scene
echo                              The setup tax is amortized here.
echo
echo     [2] RENDER ONE FRAME   a single hero frame
echo                              Shading fewer AOVs is the lever here.
echo
echo     [3] COMPUTE            GEMM / tensor throughput
echo                              This is the workload that reaches the
echo                              card's power ceiling on this box.
echo
echo     [4] PROFILE ONLY       what is this machine, and what is
echo                              limiting it right now?  (no levers run)
echo
echo     [5] SPECS              the machine's own numbers, in detail
echo
echo     [Q] Quit
echo.
set "choice="
set /p "choice=   Choose a request:  "
if /i "!choice!"=="1" (
    call :run --request shot
    goto :done
)
if /i "!choice!"=="2" (
    call :run --request render
    goto :done
)
if /i "!choice!"=="3" (
    call :run --request compute
    goto :done
)
if /i "!choice!"=="4" (
    call :run --request profile
    goto :done
)
if /i "!choice!"=="5" (
    call :specs
    goto :menu
)
if /i "!choice!"=="Q" goto :end
goto :menu

:specs
cls
echo.
echo   THIS MACHINE
echo   ------------------------------------------------------------
"%PY%" -c "import sys;sys.path.insert(0,r'%REPO%\OptimizerV12');from v12 import truth;t=truth.gpu_truth();print();print(t.summary);print('HAGS:',t.hags_enabled);print('power ceiling: %.0f W (allowed %.0f-%.0f W)'%(t.power_limit_w,t.power_limit_min_w,t.power_limit_max_w));print('memory bandwidth: %.0f GB/s'%t.mem_bandwidth_gbs);print('max SM clock: %d MHz (rated %d)'%(t.clock_sm_max_mhz,t.clock_sm_rated_mhz));print();r=truth.roofline();print(r.summary)"
echo.
echo   The CPU is not in GPU terms, so it is read separately:
"%PY%" -c "import psutil;print('  CPU cores: %d physical / %d logical'%(psutil.cpu_count(logical=False),psutil.cpu_count()));print('  RAM: %.0f GiB'%(psutil.virtual_memory().total/2**30))"
echo.
pause
goto :menu

:usage
echo   Optimize-The-4090.bat --request render^|shot^|compute^|profile
echo                       [--usd PATH] [--settings PRIM] [--frames N]
echo                       [--start-frame N] [--samples N] [--res-scale N]
echo                       [--reps N] [--warmup N] [--variant beauty_only^|no_crypto]
echo                       [--levers a,b] [--threads N] [--probe]
echo                       [--quiet-timeout SECONDS]
echo.
echo   With no arguments it asks which request you want.
echo.
echo   --request defaults to render.
echo.
echo   Every campaign commits its measurement records to the repo afterwards,
echo   so the copies under measurements\ stay current. Only measurements\
echo   is committed - any source edit you have in flight is left alone.
echo.
echo   Levers: frames_per_process, aov_variant, power_ceiling,
echo            husk_threads, manual
echo.
pause
goto :end

:done
echo.
echo   ------------------------------------------------------------
echo   done. any key returns to the menu...
pause >nul
goto :menu

:end
endlocal
