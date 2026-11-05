"""hython: write a flattened variant of the hero USD with one scene change, for A/B.

    hython harness\\variant_usd.py --usd hero.usd --settings /Render/X --out hero_beauty.usd --mode beauty_only
    modes:  beauty_only   keep only the beauty RenderVar on every product of the settings prim
            no_crypto     drop RenderVars whose name contains 'Crypto'
"""
import argparse
from pathlib import Path

from pxr import Usd, UsdRender

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--settings", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mode", choices=["beauty_only", "no_crypto"], required=True)
a = ap.parse_args()

stage = Usd.Stage.Open(a.usd)
rs = UsdRender.Settings(stage.GetPrimAtPath(a.settings))
if not rs:
    raise SystemExit(f"no RenderSettings at {a.settings}")
for ppath in rs.GetProductsRel().GetTargets():
    prod = UsdRender.Product(stage.GetPrimAtPath(ppath))
    rel = prod.GetOrderedVarsRel()
    before = [str(t) for t in rel.GetTargets()]
    if a.mode == "beauty_only":
        keep = [t for t in before if t.rsplit("/", 1)[-1] == "beauty"] or before[:1]
    else:
        keep = [t for t in before if "crypto" not in t.lower()]
    rel.SetTargets(keep)
    print(f"{ppath}: {len(before)} -> {len(keep)} vars: {[k.rsplit('/',1)[-1] for k in keep]}")
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
stage.Flatten().Export(str(out))
print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")
