@echo off
REM ===========================================================================
REM  THE METHOD  -  RETIRED 2026-11-04.  Delegates to Optimize-The-4090.bat
REM
REM  The five moves did not change. The way the numbers were compared did, and
REM  that is why this launcher now hands over instead of running.
REM
REM  Why it was retired:
REM
REM    Move 03 gated a candidate against a baseline measured in an EARLIER
REM    SESSION. Any drift between the two sessions — a different set of
REM    resident apps, a different clock state, a warmer page cache — was
REM    charged to the candidate.
REM
REM    On 2026-11-03 that produced a verdict of ADMIT +12.34% (p=1.9e-05) for
REM    a thread-count change that was worth +0.7% when the baseline was re-timed
REM    the same way as the candidate. The Field Notes call it "the move that
REM    evaporated". The verdict sat in the ledger.
REM
REM    No statistical gate can catch that on its own: Welch's t-test on two
REM    INDEPENDENT series cannot tell drift from effect. The fix is in the
REM    measurement, not the arithmetic — both arms are now measured ALTERNATELY
REM    in one session, and the baseline's drift across the run is refused if it
REM    exceeds 5%.
REM
REM  Everything this launcher did is still available, and done correctly:
REM
REM    Optimize-The-4090.bat --request render|shot|compute|profile
REM
REM  Arguments are passed straight through.
REM ===========================================================================
setlocal
title THE METHOD - retired, delegating to Optimize-The-4090
color 0F
chcp 65001 >nul

set "REPO=C:\Users\User\OneDrive\Desktop\Optimizer"

if "%~1"=="" (
    echo.
    echo   THE METHOD has been retired into Optimize-The-4090.bat.
    echo.
    echo   The five moves are unchanged. What changed is how the baseline and the
    echo   candidate are compared: the old launcher gated a candidate against a
    echo   baseline measured in an EARLIER SESSION, so drift was charged to the
    echo   candidate. On 2026-11-03 that produced a +12.34%% "win" for a change
    echo   worth +0.7%%, and the verdict was written to the ledger.
    echo.
    echo   Both arms are now measured alternately in one session, and a baseline
    echo   that drifts more than 5%% during the run is refused.
    echo.
    echo   Delegating to Optimize-The-4090.bat.
    echo.
    timeout /t 4 >nul
)

cd /d "%REPO%"
call "%REPO%\Optimize-The-4090.bat" %*
endlocal
