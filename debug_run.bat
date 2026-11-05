@echo off
echo [DEBUG] Starting Debug Run...
echo [DEBUG] Enabling verbose output...

REM Run the main script
call optimize_rtx4090_vfx.bat

echo.
echo [DEBUG] Main script finished.
echo [DEBUG] If you see this, the window did not close immediately.
echo [DEBUG] Check output above for errors.
pause
