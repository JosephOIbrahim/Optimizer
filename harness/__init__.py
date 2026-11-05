"""
harness — produces the samples V12 has been refusing to act without.

Not an optimizer. Each module times ONE fixed workload N times under a known
GPU state, records what fraction of the run sat at the power cap, and prints
the exact `v12 baseline --record` line that turns those numbers into a
recorded reference, along with the procedure for comparing a change against
it. Nothing here mutates the machine.

The candidate line it used to print — `v12 run --candidate-for "KEY=..."` — is
GONE, and that is the point. That command compares a pasted series against
whatever baseline happens to be on disk, which is usually one from an earlier
session; on 2026-11-03 it produced an ADMIT of +12.34% for a change worth
nothing. V12 now refuses a comparison whose two arms cannot be shown to come
from one session, so the printed guidance is the procedure that CAN produce
one: measure both arms alternately, in a single session.

    throttle.py     NVML clock-event counters before/after a run -> % at power cap
    husk_frame.py   one Karma XPU frame via husk, render_s + ssim guardrail
    gemm.py         8192^3 GEMM fp16/bf16/tf32 with fp32 accumulate -> tflops
    comfy_run.py    one fixed-seed ComfyUI workflow, frame_ms + lpips/clip guardrails
    common.py       measurement root (E:\\OptimizerV12\\measurements), preflight, paste lines
"""
__version__ = "0.1.0"
