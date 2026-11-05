"""
export_hero — hython: open a .hip(lc) from disk, flatten the display-flagged LOP,
write one USD for husk_frame.py. Reads the SAVED file; unsaved edits in an open
Houdini session are not in the export.

    hython harness\\export_hero.py --hip "...\\scene.hiplc" --out "E:\\OptimizerV12\\scenes\\hero.usd" [--lop /stage/karma1] [--frame 1]

Prints: the LOP used, cameras found, render-settings prims, and the camera to
pass to husk (--camera). Writes <out>.json beside the USD with the same facts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lop", default="", help="LOP node path; default = display-flagged node in /stage")
    ap.add_argument("--frame", type=float, default=None)
    args = ap.parse_args()

    import hou  # hython only
    from pxr import Usd, UsdGeom, UsdRender

    hou.hipFile.load(args.hip, suppress_save_prompt=True, ignore_load_warnings=True)
    if args.frame is not None:
        hou.setFrame(args.frame)
    stage_net = hou.node("/stage")
    if stage_net is None:
        print("no /stage network in this file")
        return 2
    if args.lop:
        lop = hou.node(args.lop)
    else:
        lop = next((n for n in stage_net.children() if n.isDisplayFlagSet()), None)
    if lop is None:
        print("no LOP found; candidates:", [n.path() for n in stage_net.children()])
        return 2
    print("LOP:", lop.path(), "frame:", hou.frame())

    stage = lop.stage()
    if stage is None:
        print("LOP produced no stage (cook error?):", lop.errors())
        return 3
    cams = [p.GetPath().pathString for p in stage.Traverse() if p.IsA(UsdGeom.Camera)]
    rs = [p.GetPath().pathString for p in stage.Traverse() if p.IsA(UsdRender.Settings)]
    hero = next((c for c in cams if "HERO" in c.upper()), cams[0] if cams else "")
    print("cameras:", cams)
    print("render settings:", rs)
    print("hero camera:", hero or "(none found)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    flat = stage.Flatten()
    ok = flat.Export(str(out))
    if not ok:
        print("export failed")
        return 4
    meta = {"hip": args.hip, "lop": lop.path(), "frame": hou.frame(), "cameras": cams,
            "render_settings": rs, "hero_camera": hero, "usd": str(out),
            "houdini": hou.applicationVersionString()}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print("husk camera flag:", f'--camera "{hero}"' if hero else "(none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
