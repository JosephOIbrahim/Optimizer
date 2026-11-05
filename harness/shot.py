"""
shot — render a SHOT (N frames of one scene) and report what it cost.

This is the measurement the Field Notes (No. 01) said was missing.

    A 3-frame shot of the hero scene, measured on this box on 2026-11-04 at
    1024x429, 128 spp, /Render/OVOID_CPU_RENDER_V013:

        3 frames, one husk process each   164.2 s   (54.7 s per frame)
        3 frames, one husk process total   88.2 s   (29.4 s per frame)

    A 46% cut, and NOT ONE SECOND of it appears in the per-frame render
    number the existing objective measures. Per-frame render_s was 48.1 s in
    both cases, because render_s is the shading phase with setup excluded and
    the setup is what changed.

    The setup tax is ~38.7 s per launch: shader compile, texture load, BVH
    build. Pay it once for three frames instead of three times and the shot
    gets 46% cheaper. This is the lever the notes called "frames per
    process", and it is off the GPU entirely.

Why the shot and not the frame
------------------------------
A frame is not what a VFX artist delivers. A shot is. And the two workloads
have different economics: a frame's cost is dominated by a fixed per-launch
tax, so the number that matters — "how long until this shot is done" — is not
visible anywhere in a per-frame metric.

So `shot_s` is the primary metric here: wall seconds to deliver the whole
shot, setup included. The per-frame render time is still recorded, because it
answers a different question ("did the shading get faster?") and the two
disagree exactly when setup is the thing that changed.

    python harness\\shot.py --usd <scene> --frames 3 --reps 3
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import common, throttle  # noqa: E402

HUSK_DEFAULT = r"C:\Program Files\Side Effects Software\Houdini 22.0.466\bin\husk.exe"

#: husk prints "Total Wall Clock Time: H:MM:SS.ss" once per frame, CUMULATIVE
#: across the process (verified on this box: 54.70, 1:10.73, 1:26.39 for three
#: frames). So the per-frame times are the DIFFERENCES, not the values.
_TOTAL_RE = re.compile(r"Total Wall Clock Time:\s*(\d+):(\d+):([\d.]+)")
_LOAD_RE = re.compile(r"Load Wall Clock Time:\s*(\d+):(\d+):([\d.]+)")
_UPDATE_RE = re.compile(r"Update Wall Clock Time:\s*(\d+):(\d+):([\d.]+)")
_RENDER_RE = re.compile(r">>> Render .*?\((\d+) of (\d+)\) - (\d+)x(\d+)")


def _hms(h: str, m: str, s: str) -> float:
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_frame_times(log: str) -> Tuple[List[float], List[float], List[str]]:
    """(per-frame wall seconds, per-frame render seconds, resolutions).

    husk's "Total Wall Clock Time" is cumulative across frames, so the
    per-frame wall time is the difference between consecutive readings. The
    per-frame RENDER time is total minus load minus update, matching what
    husk_frame.py measures for a single frame — so the two harnesses agree on
    what a frame's shading costs, and differ only in what they pay for
    getting to it.
    """
    totals = [_hms(*m.groups()) for m in _TOTAL_RE.finditer(log)]
    loads = [_hms(*m.groups()) for m in _LOAD_RE.finditer(log)]
    updates = [_hms(*m.groups()) for m in _UPDATE_RE.finditer(log)]
    res = [f"{m.group(3)}x{m.group(4)}" for m in _RENDER_RE.finditer(log)]

    wall: List[float] = []
    prev = 0.0
    for t in totals:
        wall.append(max(0.0, t - prev))
        prev = t

    render: List[float] = []
    for i, t in enumerate(totals):
        lo = loads[i] if i < len(loads) else 0.0
        up = updates[i] if i < len(updates) else 0.0
        render.append(max(0.0, t - lo - up))
    return wall, render, res


def render_shot(usd: str, frames: int, start_frame: int = 503,
                settings: str = "", pixel_samples: int = 128, res_scale: int = 100,
                renderer: str = "BRAY_HdKarmaXPU", husk: str = HUSK_DEFAULT,
                out_dir: Optional[str] = None, label: str = "shot",
                threads: int = 0) -> Dict[str, object]:
    """Render `frames` frames in ONE husk process. Returns the measurement."""
    usd_path = Path(usd)
    if not usd_path.exists():
        raise SystemExit(f"no such USD: {usd}")
    if not Path(husk).exists():
        raise SystemExit(f"no husk at {husk}")

    out = Path(out_dir or (common.MEASURE_ROOT / "husk_shots"))
    out.mkdir(parents=True, exist_ok=True)
    tag = time.strftime("%Y%m%d_%H%M%S")

    # husk requires a frame variable in the output path when rendering more
    # than one frame, or it refuses to overwrite. $F4 is the 4-digit frame.
    img = out / f"{tag}_{label}_$F4.png"
    cmd = [husk, "--usd-input", str(usd_path), "-R", renderer,
           "-f", str(start_frame), "-n", str(frames),
           "-p", str(pixel_samples), "-S", str(res_scale),
           "-o", str(img), "-V", "2"]
    if settings:
        cmd += ["-s", settings]
    if threads:
        cmd += ["-j", str(threads)]

    snap_a = throttle.Snapshot.take()
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    snap_b = throttle.Snapshot.take()
    if proc.returncode != 0:
        raise SystemExit(f"husk failed (exit {proc.returncode}):\n{proc.stderr[-1500:]}")

    frame_wall, frame_render, res = parse_frame_times(proc.stdout)
    images = sorted(str(p) for p in out.glob(f"{tag}_{label}_*.png"))

    return {
        "label": label,
        "wall_s": wall,
        "frames": frames,
        "start_frame": start_frame,
        "frame_wall_s": frame_wall,
        "frame_render_s": frame_render,
        "resolutions": res,
        "images": images,
        "throttle": throttle.delta(snap_a, snap_b),
        "settings": settings,
        "pixel_samples": pixel_samples,
        "res_scale": res_scale,
        "renderer": renderer,
        "usd": str(usd_path),
    }


def render_frames_separately(usd: str, frames: int, start_frame: int = 503,
                             settings: str = "", pixel_samples: int = 128,
                             res_scale: int = 100, renderer: str = "BRAY_HdKarmaXPU",
                             husk: str = HUSK_DEFAULT, out_dir: Optional[str] = None,
                             label: str = "shot_split", threads: int = 0
                             ) -> Dict[str, object]:
    """The SAME shot, rendered one frame per process. The control arm.

    This is how the work is done today, and it is the arm the frames-per-
    process lever is measured against. Measuring both in the same session,
    alternating, is what makes the comparison trustworthy.
    """
    usd_path = Path(usd)
    out = Path(out_dir or (common.MEASURE_ROOT / "husk_shots"))
    out.mkdir(parents=True, exist_ok=True)
    tag = time.strftime("%Y%m%d_%H%M%S")

    snap_a = throttle.Snapshot.take()
    t0 = time.perf_counter()
    per_frame_wall: List[float] = []
    per_frame_render: List[float] = []
    images: List[str] = []
    for i in range(frames):
        f = start_frame + i
        img = out / f"{tag}_{label}_{f:04d}.png"
        cmd = [husk, "--usd-input", str(usd_path), "-R", renderer,
               "-f", str(f), "-n", "1", "-p", str(pixel_samples),
               "-S", str(res_scale), "-o", str(img), "-V", "2"]
        if settings:
            cmd += ["-s", settings]
        if threads:
            cmd += ["-j", str(threads)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise SystemExit(f"husk failed on frame {f} (exit {p.returncode}):\n"
                             f"{p.stderr[-1500:]}")
        w, r, _ = parse_frame_times(p.stdout)
        per_frame_wall.append(w[-1] if w else 0.0)
        per_frame_render.append(r[-1] if r else 0.0)
        if img.exists():
            images.append(str(img))
    wall = time.perf_counter() - t0
    snap_b = throttle.Snapshot.take()

    return {
        "label": label,
        "wall_s": wall,
        "frames": frames,
        "start_frame": start_frame,
        "frame_wall_s": per_frame_wall,
        "frame_render_s": per_frame_render,
        "resolutions": [],
        "images": images,
        "throttle": throttle.delta(snap_a, snap_b),
        "settings": settings,
        "pixel_samples": pixel_samples,
        "res_scale": res_scale,
        "renderer": renderer,
        "usd": str(usd_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usd", required=True)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--start-frame", type=int, default=503)
    ap.add_argument("--settings", default="")
    ap.add_argument("--pixel-samples", type=int, default=128)
    ap.add_argument("--res-scale", type=int, default=100)
    ap.add_argument("--renderer", default="BRAY_HdKarmaXPU")
    ap.add_argument("--husk", default=HUSK_DEFAULT)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--split", action="store_true",
                    help="render one frame per process instead of one process for the shot")
    ap.add_argument("--label", default="shot")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    if not args.skip_preflight and not common.preflight():
        return 1

    fn = render_frames_separately if args.split else render_shot
    walls: List[float] = []
    for i in range(args.reps):
        r = fn(args.usd, args.frames, args.start_frame, args.settings,
               args.pixel_samples, args.res_scale, args.renderer, args.husk,
               label=f"{args.label}{i}")
        walls.append(float(r["wall_s"]))
        print(f"rep {i}: {r['wall_s']:.2f} s wall  "
              f"frames {[round(t, 2) for t in r['frame_wall_s']]}")
    s = common.summarize(walls)
    print(f"\nshot_s: mean {s['mean']:.2f}  cv {s.get('cv', 0):.2%}  n={s['n']:.0f}")
    rec = common.write_measurement("shot", {
        "harness": "shot", "label": args.label, "mode": "split" if args.split else "batched",
        "usd": args.usd, "frames": args.frames, "settings": args.settings,
        "pixel_samples": args.pixel_samples, "res_scale": args.res_scale,
        "shot_s": walls, "summary": s,
    })
    print(f"saved {rec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
