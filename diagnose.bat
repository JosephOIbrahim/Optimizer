@echo off
echo [DIAGNOSIS] Starting diagnosis...
echo [DIAGNOSIS] Logging to diagnosis.log...

echo ========================================== > diagnosis.log
echo DATE: %DATE% %TIME% >> diagnosis.log
echo CD: %CD% >> diagnosis.log
echo DP0: %~dp0 >> diagnosis.log
echo ========================================== >> diagnosis.log

echo [DIAGNOSIS] Checking Python...
python --version >> diagnosis.log 2>&1
if %errorLevel% == 0 (
    echo Python found. >> diagnosis.log
) else (
    echo Python NOT found. >> diagnosis.log
)

echo [DIAGNOSIS] Checking Admin privileges...
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Admin: YES >> diagnosis.log
) else (
    echo Admin: NO >> diagnosis.log
)

echo [DIAGNOSIS] Checking venv...
if exist "venv\" (
    echo venv exists. >> diagnosis.log
) else (
    echo venv missing. >> diagnosis.log
)

echo [DIAGNOSIS] Done.
echo [DIAGNOSIS] Please check diagnosis.log
pause
