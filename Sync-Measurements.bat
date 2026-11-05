@echo off
REM ===========================================================================
REM  Sync-Measurements - copy the live measurement records into the repo and
REM  commit them, so the committed copies never go stale after a run.
REM
REM  The live store is E:\OptimizerV12\measurements (off OneDrive on purpose).
REM  The repo keeps a copy under measurements\, committed, so a figure quoted
REM  in the operator cards can be checked against the record it came from.
REM
REM  Run this AFTER a measurement campaign.
REM ===========================================================================
setlocal enabledelayedexpansion
title Sync Measurements - Optimizer repo
color 0B
chcp 65001 >nul

set "REPO=C:\Users\User\OneDrive\Desktop\Optimizer"
set "V12PY=C:\Python314\python.exe"
cd /d "%REPO%"
set "PYTHONUNBUFFERED=1"

if not exist "%V12PY%" (echo [ERROR] %V12PY% missing & pause & exit /b 1)
if not exist "%REPO%\harness\sync_measurements.py" (echo [ERROR] harness\sync_measurements.py missing & pause & exit /b 1)

if /i "%~1"=="--check" goto check

"%V12PY%" harness\sync_measurements.py %*
set RC=%ERRORLEVEL%
echo.
if %RC%==0 (echo [OK] done) else (echo [FAILED] exit %RC%)
pause
exit /b %RC%

:check
echo.
echo --- what would change, if anything ---
"%V12PY%" harness\sync_measurements.py --check
echo.
pause
exit /b 0
