"""
selftest — prove the harness plumbing on THIS machine before trusting a number.

    C:\\Python314\\python.exe harness\\selftest.py

1. Throttle counters: runs a ~6 s fp16 load in the torch venv and reads the
   SW-power-cap counter through BOTH the NVML path and nvidia-smi. They must
   agree in units; if nvidia-smi moves and NVML does not, the NVML path is wrong.
2. husk: renders a generated sphere scene (16 spp, 25 % res) to prove the flag set.
Nothing here is a baseline. Nothing is applied.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import common, throttle  # noqa: E402

TORCHPY = r"E:\OptimizerV12\venv\Scripts\python.exe"
HUSK = r"C:\Program Files\Side Effects Software\Houdini 22.0.466\bin\husk.exe"
LOAD = ("import torch,time\n"
        "a=torch.randn(8192,8192,device='cuda',dtype=torch.float16);b=a.clone()\n"
        "t=time.time()\n"
        "while time.time()-t<6: torch.matmul(a,b)\n"
        "torch.cuda.synchronize()\n")
SPHERE = """#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
)
def Xform "World"
{
    def Sphere "ball"
    {
        double radius = 1
    }
    def Camera "cam"
    {
        float focalLength = 35
        double3 xformOp:translate = (0, 0, 6)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
"""


def smi_power_us() -> int:
    out = subprocess.run(["nvidia-smi", "-q", "-d", "PERFORMANCE"], capture_output=True, text=True).stdout
    m = re.search(r"SW Power Capping\s*:\s*(\d+) us", out)
    return int(m.group(1)) if m else -1


def test_throttle() -> bool:
    print("== throttle counters: NVML vs nvidia-smi around a 6 s fp16 load")
    s0 = smi_power_us()
    a = throttle.Snapshot.take()
    r = subprocess.run([TORCHPY, "-c", LOAD], capture_output=True, text=True)
    b = throttle.Snapshot.take()
    s1 = smi_power_us()
    if r.returncode != 0:
        print("   load failed:", r.stderr[-400:])
        return False
    nv = b.violation_us.get("power", 0) - a.violation_us.get("power", 0)
    wall = b.t_wall - a.t_wall
    d = throttle.delta(a, b)
    print(f"   wall {wall:.2f} s   smi counter delta {s1 - s0} us   nvml raw delta {nv}   source {a.source}")
    print(f"   harness says: {throttle.describe(d)}")
    print(f"   smi says:     at power cap {(s1 - s0) / 1e6 / wall:.0%}")
    if s1 - s0 > 0 and nv == 0:
        print("   MISMATCH: nvidia-smi moved, NVML path did not. Falling back to nvidia-smi is safe; NVML path needs a look.")
        return False
    if s1 - s0 > 0 and nv > 0:
        ratio = nv / (s1 - s0)
        print(f"   nvml/smi ratio {ratio:.1f}  (1000 = NVML in ns as assumed; 1 = NVML in us -> fix unit)")
    return True


def test_husk() -> bool:
    print("== husk flag set on a generated sphere")
    d = common.MEASURE_ROOT / "smoke"
    d.mkdir(parents=True, exist_ok=True)
    usd = d / "sphere.usda"
    usd.write_text(SPHERE, encoding="ascii", newline="\n")
    out = d / "sphere.png"
    cmd = [HUSK, "--usd-input", str(usd), "-R", "BRAY_HdKarmaXPU", "-f", "1", "-n", "1",
           "-p", "16", "-S", "25", "-o", str(out), "-V", "1"]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    t = time.perf_counter() - t0
    if r.returncode != 0 or not out.exists():
        print(f"   husk exit {r.returncode} in {t:.1f}s")
        print("   stdout:", r.stdout[-800:])
        print("   stderr:", r.stderr[-800:])
        return False
    print(f"   rendered {out.name} in {t:.1f} s")
    return True


if __name__ == "__main__":
    ok = test_throttle()
    ok = test_husk() and ok
    print("\nselftest:", "PASS" if ok else "FAIL (see above)")
    raise SystemExit(0 if ok else 1)
