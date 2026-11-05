@echo off
REM ===========================================================================
REM  Measure-This-Machine  -  produces the samples V12 needs. Applies NOTHING.
REM  Lives in the Optimizer repo. Measurements -> E:\OptimizerV12\measurements
REM ===========================================================================
setlocal enabledelayedexpansion
title Measure This Machine - RTX 4090 harness
color 0B
chcp 65001 >nul

set "REPO=C:\Users\User\OneDrive\Desktop\Optimizer"
set "V12PY=C:\Python314\python.exe"
set "TORCHPY=E:\OptimizerV12\venv\Scripts\python.exe"
set "CFG=E:\OptimizerV12\harness_config.json"
cd /d "%REPO%"
set "PYTHONUNBUFFERED=1"
set "RENDERPY=%V12PY%"
if exist "%TORCHPY%" set "RENDERPY=%TORCHPY%"

if not exist "%V12PY%" (echo [ERROR] %V12PY% missing - V12 preflight needs it & pause & exit /b 1)
if not exist "E:\OptimizerV12" mkdir "E:\OptimizerV12"

:menu
cls
echo ================================================================
echo        MEASURE THIS MACHINE   -   harness for Optimizer V12
echo        every option is read-only on the GPU; nothing is applied
echo ================================================================
echo.
echo     [7]  Preflight         is anything heavy on the GPU?  (must be CLEAR)
echo     [1]  Render            one Karma XPU frame via husk  (render_s + ssim)   hero USD: E:\OptimizerV12\scenes\ovoid_hero_v002.usd
echo     [2]  GEMM              8192^3 fp16/bf16/tf32/fp32     (tflops; needs torch venv)
echo     [3]  Diffusion         fixed-seed ComfyUI workflow    (frame_ms + lpips/clip)
echo     [4]  Throttle watch    10 s of power-cap counters while YOU run something
echo     [5]  Torch venv        status of E:\OptimizerV12\venv
echo     [6]  Open measurements folder
echo.
echo     [Q]  Quit
echo.
set "choice="
set /p "choice=   Select:  "
if /i "!choice!"=="7" ( "%V12PY%" -m v12 preflight & cd /d "%REPO%" & goto pauseback )
if /i "!choice!"=="1" ( call :render & goto pauseback )
if /i "!choice!"=="2" ( call :gemm & goto pauseback )
if /i "!choice!"=="3" ( call :diffusion & goto pauseback )
if /i "!choice!"=="4" ( "%V12PY%" harness\throttle.py 10 & goto pauseback )
if /i "!choice!"=="5" ( call :venvstatus & goto pauseback )
if /i "!choice!"=="6" ( start "" "E:\OptimizerV12\measurements" & goto menu )
if /i "!choice!"=="Q" goto end
goto menu

:render
set "USD="
if exist "%CFG%" for /f "usebackq delims=" %%u in (`"%V12PY%" -c "import json;print(json.load(open(r'%CFG%')).get('usd',''))"`) do set "USD=%%u"
if defined USD echo   last USD: !USD!
set "in="
set /p "in=   USD file for the hero frame [Enter = last]:  "
if not "!in!"=="" set "USD=!in!"
if not defined USD (echo   no USD given. Export the hero frame from Solaris once ^(USD ROP / File ^> Export^). & exit /b)
set "FRAME=1"
set /p "FRAME=   frame [1]:  "
set "RS=/Render/OVOID_CPU_RENDER_V013"
set /p "RS=   RenderSettings prim (pins res+camera) [/Render/OVOID_CPU_RENDER_V013]:  "
set "LABEL=baseline"
set /p "LABEL=   label (baseline / the change you applied) [baseline]:  "
"%V12PY%" -c "import json,os;p=r'%CFG%';d=json.load(open(p)) if os.path.exists(p) else {};d['usd']=r'!USD!';json.dump(d,open(p,'w'),indent=2)"
echo.
"%RENDERPY%" harness\husk_frame.py --usd "!USD!" --frame !FRAME! --settings "!RS!" --label "!LABEL!"
exit /b

:gemm
if not exist "%TORCHPY%" (echo   torch venv not ready yet - see option [5] & exit /b)
set "LABEL=baseline"
set /p "LABEL=   label [baseline]:  "
"%TORCHPY%" harness\gemm.py --label "!LABEL!"
exit /b

:diffusion
if not exist "%TORCHPY%" (echo   torch venv not ready yet - see option [5] & exit /b)
set "WF="
if exist "%CFG%" for /f "usebackq delims=" %%u in (`"%V12PY%" -c "import json;print(json.load(open(r'%CFG%')).get('workflow',''))"`) do set "WF=%%u"
if defined WF echo   last workflow: !WF!
set "in="
set /p "in=   API-format workflow JSON (fixed seed) [Enter = last]:  "
if not "!in!"=="" set "WF=!in!"
if not defined WF (echo   no workflow given. & exit /b)
set "REF="
set /p "REF=   reference PNG from the baseline run (blank = this IS the baseline):  "
set "LABEL=baseline"
set /p "LABEL=   label [baseline]:  "
"%V12PY%" -c "import json,os;p=r'%CFG%';d=json.load(open(p)) if os.path.exists(p) else {};d['workflow']=r'!WF!';json.dump(d,open(p,'w'),indent=2)"
echo   (ComfyUI must already be running - launch it the way you want measured)
if defined REF ( "%TORCHPY%" harness\comfy_run.py --workflow "!WF!" --reference "!REF!" --label "!LABEL!" ) else ( "%TORCHPY%" harness\comfy_run.py --workflow "!WF!" --label "!LABEL!" )
exit /b

:venvstatus
if exist "%TORCHPY%" (
  "%TORCHPY%" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'device ok',torch.cuda.is_available())" 2>nul || echo   venv exists but torch not importable yet
) else echo   venv not created yet
if exist "E:\OptimizerV12\logs\venv_setup.log" (echo   --- last lines of setup log --- & powershell -NoProfile -Command "Get-Content 'E:\OptimizerV12\logs\venv_setup.log' -Tail 6")
exit /b

:pauseback
echo.
echo ----------------------------------------------------------------
echo   done. any key returns to the menu...
pause >nul
goto menu

:end
endlocal
