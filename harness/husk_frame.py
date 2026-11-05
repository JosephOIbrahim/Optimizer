"""
husk_frame — time ONE Karma XPU frame N times. The render objective in V12.

    python husk_frame.py --usd "E:\\path\\hero.usd" --frame 1001 --reps 6

Fixed per run: renderer, frame, pixel samples, resolution scale, output format.
Changing any of them changes the workload fingerprint V12 compares against, so
pass the same flags for baseline and candidate. The first rep is a warm-up
(shader compile, texture load) and is discarded; V12 needs >= 5 real samples.

Primary:   render_s   (wall seconds of husk, lower is better)
Guardrail: ssim       (vs the first measured rep; a change that alters the
                       image is not a speedup — e.g. a sample-count "win")
Throttle:  % of each rep at the power cap, from harness.throttle
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import common, throttle  # noqa: E402

HUSK_DEFAULT = r"C:\Program Files\Side Effects Software\Houdini 22.0.466\bin\husk.exe"


def ssim_png(a: Path, b: Path):
    """Global SSIM on luminance, 8x8 uniform windows. No scipy. Returns None
    when PIL/numpy are missing so the run still records render_s."""
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return None
    x = np.asarray(Image.open(a).convert("L"), dtype=np.float64)
    y = np.asarray(Image.open(b).convert("L"), dtype=np.float64)
    if x.shape != y.shape:
        return None
    h, w = (x.shape[0] // 8) * 8, (x.shape[1] // 8) * 8
    x, y = x[:h, :w], y[:h, :w]
    bx = x.reshape(h // 8, 8, w // 8, 8).mean(axis=(1, 3))
    by = y.reshape(h // 8, 8, w // 8, 8).mean(axis=(1, 3))
    vx = x.reshape(h // 8, 8, w // 8, 8).var(axis=(1, 3))
    vy = y.reshape(h // 8, 8, w // 8, 8).var(axis=(1, 3))
    cov = ((x.reshape(h // 8, 8, w // 8, 8) - bx[:, None, :, None])
           * (y.reshape(h // 8, 8, w // 8, 8) - by[:, None, :, None])).mean(axis=(1, 3))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    s = ((2 * bx * by + c1) * (2 * cov + c2)) / ((bx ** 2 + by ** 2 + c1) * (vx + vy + c2))
    return float(s.mean())


def husk_times(log: str) -> dict:
    """Pull husk's own phase timers out of -V 1 output (H:MM:SS.ss)."""
    import re
    out = {}
    for key, pat in (("load", r"Load Wall Clock Time:\s*(\d+):(\d+):([\d.]+)"),
                     ("update", r"Update Wall Clock Time:\s*(\d+):(\d+):([\d.]+)"),
                     ("total", r"Total Wall Clock Time:\s*(\d+):(\d+):([\d.]+)")):
        m = re.search(pat, log)
        if m:
            h, mi, sec = m.groups()
            out[key] = int(h) * 3600 + int(mi) * 60 + float(sec)
    m = re.search(r">>> Render .*? - (\d+x\d+)", log)
    if m:
        out["res"] = m.group(1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usd", required=True)
    ap.add_argument("--frame", type=int, default=1)
    ap.add_argument("--reps", type=int, default=6, help="incl. 1 warm-up (default 6 -> 5 samples)")
    ap.add_argument("--pixel-samples", type=int, default=128)
    ap.add_argument("--res-scale", type=int, default=100)
    ap.add_argument("--renderer", default="BRAY_HdKarmaXPU")
    ap.add_argument("--camera", default="", help="USD camera prim path (from export_hero.py)")
    ap.add_argument("--threads", type=int, default=0, help="husk -j (0 = all cores)")
    ap.add_argument("--settings", default="", help="RenderSettings prim, e.g. /Render/OVOID_CPU_RENDER_V013 (pins res + camera)")
    ap.add_argument("--husk", default=HUSK_DEFAULT)
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--label", default="baseline", help="tag written into the record")
    args = ap.parse_args()

    usd = Path(args.usd)
    if not usd.exists():
        print(f"no such USD: {usd}")
        return 2
    if not Path(args.husk).exists():
        print(f"no husk at {args.husk}")
        return 2
    if not args.skip_preflight and not common.preflight():
        return 1

    common.ensure_dirs()
    out_dir = common.MEASURE_ROOT / "husk_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = time.strftime("%Y%m%d_%H%M%S")

    print(f"\nhusk {args.renderer}  {usd.name}  f{args.frame}  {args.pixel_samples}spp  settings={args.settings or '-'}  "
          f"res {args.res_scale}%  x{args.reps} (first is warm-up)\n")
    render_s, process_s, ssims, throttles, images = [], [], [], [], []
    for i in range(args.reps):
        img = out_dir / f"{tag}_{args.label}_rep{i:02d}.png"
        cmd = [args.husk, "--usd-input", str(usd), "-R", args.renderer, "-f", str(args.frame),
               "-n", "1", "-p", str(args.pixel_samples), "-S", str(args.res_scale),
               "-o", str(img), "-V", "2"] + (["--camera", args.camera] if args.camera else []) + (["-s", args.settings] if args.settings else []) + (["-j", str(args.threads)] if args.threads else [])
        a = throttle.Snapshot.take()
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        t = time.perf_counter() - t0
        b = throttle.Snapshot.take()
        if r.returncode != 0 or not img.exists():
            print(f"rep {i}: husk failed (exit {r.returncode})\n{r.stderr[-1500:]}")
            return 3
        d = throttle.delta(a, b)
        hk = husk_times(r.stdout)
        # render_s = husk's own total minus its load and update phases: the part a
        # GPU lever can change. Process wall (t) is kept in the record as context.
        ro = hk.get("total", t) - hk.get("load", 0.0) - hk.get("update", 0.0)
        warm = i == 0
        print(f"rep {i}: render {ro:7.2f} s  (process {t:6.1f} s, load {hk.get('load', 0):.1f}, "
              f"update {hk.get('update', 0):.1f}, {hk.get('res', '?')})  "
              f"{'warm-up, discarded' if warm else ''}  [{throttle.describe(d)}]")
        if warm:
            continue
        render_s.append(ro)
        process_s.append(t)
        throttles.append(d)
        images.append(str(img))
        if len(images) > 1:
            s = ssim_png(Path(images[0]), img)
            if s is not None:
                ssims.append(s)

    if ssims:
        ssims.insert(0, 1.0)  # the reference rep vs itself
    summ = common.summarize(render_s)
    print(f"\nrender_s: mean {summ['mean']:.3f}  median {summ['median']:.3f}  cv {summ.get('cv', 0):.2%}")
    if ssims:
        print(f"ssim vs rep1: {common.series(ssims, 4)}")
    cap = [d.get("power_frac", 0.0) for d in throttles]
    if cap:
        print(f"time at power cap per rep: {', '.join(f'{c:.0%}' for c in cap)}")

    rec = common.write_measurement("render", {
        "objective": "render", "harness": "husk_frame", "label": args.label,
        "usd": str(usd), "frame": args.frame, "renderer": args.renderer, "camera": args.camera, "threads": args.threads,
        "pixel_samples": args.pixel_samples, "res_scale": args.res_scale,
        "render_s": render_s, "process_s": process_s, "settings": args.settings, "ssim": ssims, "throttle": throttles, "images": images,
        "summary": summ,
    })
    print(f"\nsaved {rec}\n")
    for line in common.paste_lines("render", render_s, "husk_frame",
                                   {"ssim": ssims} if ssims else None):
        print(line)
    if not ssims:
        why = "fewer than 2 measured reps" if len(images) < 2 else "PIL/numpy missing in this interpreter"
        print(f"# NOTE: ssim not computed ({why}) — the render objective declares an ssim\n"
              "# guardrail, so V12 will deny until it is supplied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
