@echo off
REM ===========================================================================
REM  Sync-Measurements - copy the live measurement records into the repo and
REM  commit them, so the committed copies never go stale after a run.
REM
REM  The live store is E:\OptimizerV12\measurements (off OneDrive on purpose).
REM  The repo keeps a copy under measurements\, committed, so a figure quoted
REM  in the operator cards can be checked against the record it came from.
REM
REM  Optimize-The-4090.bat runs this automatically after every campaign, so
REM  the committed copies never go stale. Run it by hand only to refresh
REM  the copies without measuring anything.
REM
REM    --check      report what would change; copy and commit nothing
REM    /nopause    do not wait for a keypress (used when chained from another
REM                 launcher, which is already the thing the operator is
REM                 waiting on)
REM ===========================================================================
setlocal enabledelayedexpansion
title Sync Measurements - Optimizer repo
color 0B
chcp 65001 >nul

set "REPO=C:\Users\User\OneDrive\Desktop\Optimizer"
set "V12PY=C:\Python314\python.exe"
cd /d "%REPO%"
set "PYTHONUNBUFFERED=1"

set "PAUSE=1"
set "ARGS="
for %%A in (%*) do (
    if /I "%%~A"=="/nopause" (set "PAUSE=0") else (set "ARGS=!ARGS! %%~A")
)

if not exist "%V12PY%" (echo [ERROR] %V12PY% missing & goto :fail)
if not exist "%REPO%\harness\sync_measurements.py" (echo [ERROR] harness\sync_measurements.py missing & goto :fail)

if /i "%~1"=="--check" goto check

"%V12PY%" harness\sync_measurements.py !ARGS!
set "RC=!ERRORLEVEL!"
echo.
if !RC!==0 (echo [OK] done) else (echo [FAILED] exit !RC!)
if "!PAUSE!"=="1" pause
exit /b !RC!

:check
echo.
echo --- what would change, if anything ---
"%V12PY%" harness\sync_measurements.py --check
echo.
if "!PAUSE!"=="1" pause
exit /b 0

:fail
echo [ERROR] the sync could not run.
if "!PAUSE!"=="1" pause
exit /b 1
