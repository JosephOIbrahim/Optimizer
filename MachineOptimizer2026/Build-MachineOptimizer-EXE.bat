@echo off
setlocal enabledelayedexpansion
title Build MachineOptimizer 2026 .exe

REM No self-elevation here on purpose: building is an ordinary user action.
REM The .exe itself requests Administrator via an embedded manifest (--uac-admin),
REM so the built binary elevates when it is RUN, not when it is built.

cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1

echo ================================================================
echo   Build MachineOptimizer 2026  -  standalone .exe
echo ----------------------------------------------------------------
echo   Produces dist\MachineOptimizer2026.exe -- no Python needed on
echo   the target machine.
echo ================================================================
echo.

REM ==== Python =====================================================
set "PYEXE="
py -3.13 -c "" >nul 2>&1 && set "PYEXE=py -3.13"
if not defined PYEXE ( py -3.12 -c "" >nul 2>&1 && set "PYEXE=py -3.12" )
if not defined PYEXE ( python -c "" >nul 2>&1 && set "PYEXE=python" )
if not defined PYEXE (
    echo [ERROR] No Python 3.11+ found. Install from https://www.python.org/downloads/
    pause & exit /b 1
)
echo Using Python: %PYEXE%

REM ==== build environment ===========================================
REM A throwaway venv keeps PyInstaller and its dependencies out of the
REM interpreter this machine uses for real work.
if not exist ".buildenv\Scripts\python.exe" (
    echo Creating build environment .buildenv ...
    %PYEXE% -m venv .buildenv
    if errorlevel 1 ( echo [ERROR] could not create .buildenv & pause & exit /b 1 )
)
set "BPY=.buildenv\Scripts\python.exe"

"%BPY%" -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller ^(needs network, once^) ...
    "%BPY%" -m pip install --quiet --upgrade pip
    "%BPY%" -m pip install --quiet pyinstaller
    if errorlevel 1 (
        echo [ERROR] pip could not install PyInstaller -- check your network connection.
        pause & exit /b 1
    )
)

REM ==== prove the source runs BEFORE freezing it ====================
REM Freezing a package that cannot import only buries the error inside the .exe.
echo.
echo Verifying the source imports and finds its agents ...
"%BPY%" -c "from optimizer.orchestrator import discover_agents; n=len(discover_agents()); print('  agents discovered:', n); raise SystemExit(0 if n else 1)"
if errorlevel 1 (
    echo [ERROR] The source tree cannot find its agents -- fix that first.
    echo         A frozen build would inherit the same problem, silently.
    pause & exit /b 1
)

REM ==== build =======================================================
REM --collect-submodules optimizer is the load-bearing flag.
REM   discover_agents() uses pkgutil + importlib, which PyInstaller's static
REM   analysis cannot follow. Without this flag the .exe loads 0 agents, prints
REM   an empty plan, and exits 0 -- looking exactly like a clean machine.
REM --uac-admin embeds a manifest so the .exe elevates when run: it writes
REM   registry values and Defender exclusions.
echo.
echo Building ...
"%BPY%" -m PyInstaller ^
    --noconfirm --clean --onefile ^
    --name MachineOptimizer2026 ^
    --uac-admin ^
    --collect-submodules optimizer ^
    --paths . ^
    mos2026.py
if errorlevel 1 (
    echo.
    echo [ERROR] PyInstaller failed. Scroll up for the first real error.
    pause & exit /b 1
)

REM ==== ship the editable bits beside the exe =======================
REM config.toml is read from the .exe's own folder at runtime (see mos2026.py).
if exist "config.toml" copy /y "config.toml" "dist\config.toml" >nul
if exist "README.md"   copy /y "README.md"   "dist\README.md"   >nul

echo.
echo ================================================================
echo   Build complete.
echo ================================================================
for %%F in ("dist\MachineOptimizer2026.exe") do echo   exe:  %%~fF  ^(%%~zF bytes^)
echo   side: dist\config.toml  (edit this to tune the build's defaults)
echo.
echo   Run it:   dist\MachineOptimizer2026.exe --dry-run
echo             dist\MachineOptimizer2026.exe --status
echo             dist\MachineOptimizer2026.exe --watch
echo.
echo   A frozen .exe keeps its own state at the same off-OneDrive data root
echo   as the .bat launcher -- both use optimizer\paths.py, so the two are
echo   interchangeable and see the same restore points.
echo.
echo   To share it: copy the whole dist folder. No Python required.
echo ================================================================
pause
