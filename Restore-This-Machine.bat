@echo off
setlocal enabledelayedexpansion
title MachineOptimizer 2026 - Restore

net session >nul 2>&1
if %errorlevel% NEQ 0 (
    echo Requesting Administrator privileges...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

REM This launcher lives on the Desktop but drives the optimizer in its project folder.
set "OPTDIR=C:\Users\User\OneDrive\Desktop\Optimizer\MachineOptimizer2026"
cd /d "%OPTDIR%"
chcp 65001 >nul
set PYTHONUTF8=1

echo ================================================================
echo   MachineOptimizer 2026  -  RESTORE to original machine state
echo ================================================================
echo.

REM Stop any running watchdog first, else it re-applies the changes within ~20s.
echo Stopping any running watchdog...
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'orchestrator --watch' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" 2>nul

if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe -m optimizer.orchestrator --restore
) else (
    set "PYLAUNCH="
    py -3.13 -c "" >nul 2>&1 && set "PYLAUNCH=py -3.13"
    if not defined PYLAUNCH ( py -3.12 -c "" >nul 2>&1 && set "PYLAUNCH=py -3.12" )
    if not defined PYLAUNCH set "PYLAUNCH=python"
    !PYLAUNCH! -m optimizer.orchestrator --restore
)

echo.
echo Every recorded change has been reverted to its original value.
call deactivate 2>nul
pause
