@echo off
REM Optimizer V8 - GUI launcher entry point.
REM Opens launcher_ui.py via pythonw (no console window).
cd /d "%~dp0"
where pythonw >nul 2>&1
if %errorLevel% equ 0 (
    start "" pythonw "%~dp0launcher_ui.py"
) else (
    where python >nul 2>&1
    if %errorLevel% equ 0 (
        start "" python "%~dp0launcher_ui.py"
    ) else (
        echo [ERROR] No Python found on PATH. Install Python 3.10+.
        pause
    )
)
