@echo off
REM ===========================================================================
REM  Optimizer V11 desktop launcher (menu). Lives in the repo; a shortcut on
REM  C:\Users\Public\Desktop points here. Double-click -> pick an action.
REM  All menu actions are read-only compute except [7], which runs --dry-run.
REM ===========================================================================
setlocal enabledelayedexpansion
title Optimizer V11 - RTX 4090 Control Core
color 0A

set "REPO=C:\Users\User\OneDrive\Desktop\Optimizer"
if not exist "%REPO%\v9_optimizer.bat" (
    echo [ERROR] Optimizer not found at "%REPO%".
    echo         Edit the REPO path at the top of this launcher if the project moved.
    pause
    exit /b 1
)
cd /d "%REPO%"

:menu
cls
echo ================================================================
echo             OPTIMIZER V11   -   RTX 4090 Control Core
echo        hardware-modeling, self-tuning GPU optimizer
echo ================================================================
echo.
echo     [1]  Full V11 loop      roofline, analyze, autotune
echo     [2]  Doctor             honest capability report
echo     [3]  Roofline           derive + measure the 4090 ceilings
echo     [4]  Analyze            profile a workload, classify, advise
echo     [5]  Autotune           roofline-guided precision search
echo     [6]  Benchmark          measured speedup proof
echo     [7]  Optimize once      apply config tuning  (dry-run, safe)
echo.
echo     [Q]  Quit
echo.
set "choice="
set /p "choice=   Select an option:  "

if /i "!choice!"=="1" ( call :fullloop & goto pauseback )
if /i "!choice!"=="2" ( call :run doctor & goto pauseback )
if /i "!choice!"=="3" ( call :run roofline & goto pauseback )
if /i "!choice!"=="4" ( call :analyze & goto pauseback )
if /i "!choice!"=="5" ( call :run autotune & goto pauseback )
if /i "!choice!"=="6" ( call :run benchmark & goto pauseback )
if /i "!choice!"=="7" ( call :runargs once --dry-run & goto pauseback )
if /i "!choice!"=="Q" ( goto end )
goto menu

:fullloop
call :run roofline
call :runargs analyze gemm
call :runargs analyze dit
call :run autotune
exit /b

:analyze
set "wl="
set /p "wl=   Workload (gemm / elementwise / dit) [dit]:  "
if "!wl!"=="" set "wl=dit"
call :runargs analyze !wl!
exit /b

:run
echo.
echo --- v9_optimizer.bat %1 ---------------------------------------
echo.
call "%REPO%\v9_optimizer.bat" %1
exit /b

:runargs
echo.
echo --- v9_optimizer.bat %* ---------------------------------------
echo.
call "%REPO%\v9_optimizer.bat" %*
exit /b

:pauseback
echo.
echo ----------------------------------------------------------------
echo   done. press any key to return to the menu...
pause >nul
goto menu

:end
endlocal
