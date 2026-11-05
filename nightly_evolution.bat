@echo off
:: Nightly Evolution Script for Optimizer v7_gemini3
:: Runs at 3 AM to evolve system configuration

:: Project root resolved from this script's location -- survives renames.
cd /d "%~dp0"
set "PROJECT_ROOT=%~dp0"
set "VENV_PYTHON=%PROJECT_ROOT%venv\Scripts\python.exe"
set "LOG_FILE=%PROJECT_ROOT%data\nightly_evolution.log"

echo [Nightly Evolution] Starting at %DATE% %TIME% >> "%LOG_FILE%"

:: Check if Python environment exists
if not exist "%VENV_PYTHON%" (
    echo [Error] Virtual environment not found at %VENV_PYTHON% >> "%LOG_FILE%"
    exit /b 1
)

:: Run Genetic Evolution Cycle
echo [Info] Running genetic evolution... >> "%LOG_FILE%"
"%VENV_PYTHON%" "%PROJECT_ROOT%\core\gpu_optimizer_v6.py" --config config.yaml --evolve >> "%LOG_FILE%" 2>&1

if %ERRORLEVEL% EQU 0 (
    echo [Success] Evolution cycle completed successfully. >> "%LOG_FILE%"
) else (
    echo [Error] Evolution cycle failed with code %ERRORLEVEL%. >> "%LOG_FILE%"
)

echo [Nightly Evolution] Finished at %DATE% %TIME% >> "%LOG_FILE%"
echo --------------------------------------------------- >> "%LOG_FILE%"
